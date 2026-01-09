# -*- coding: utf-8 -*-
"""E5-V adapter (royokong/e5-v).

This adapter follows the official HF example **strictly** for embedding extraction:

.. code-block:: python

    processor = LlavaNextProcessor.from_pretrained('royokong/e5-v')
    model = LlavaNextForConditionalGeneration.from_pretrained(
        'royokong/e5-v', torch_dtype=torch.float16
    ).cuda()

    img_prompt  = llama3_template.format('<image>\nSummary above image in one word: ')
    text_prompt = llama3_template.format('<sent>\nSummary above sentence in one word: ')

    text_inputs = processor([text_prompt.replace('<sent>', text) for text in texts],
                            return_tensors='pt', padding=True).to('cuda')
    img_inputs  = processor([img_prompt]*len(images), images,
                            return_tensors='pt', padding=True).to('cuda')

    with torch.no_grad():
        text_embs = model(**text_inputs, output_hidden_states=True, return_dict=True)
                      .hidden_states[-1][:, -1, :]
        img_embs  = model(**img_inputs, output_hidden_states=True, return_dict=True)
                      .hidden_states[-1][:, -1, :]
        text_embs = F.normalize(text_embs, dim=-1)
        img_embs  = F.normalize(img_embs,  dim=-1)

Notes
-----
* Retrieval here is **text query -> image document** (rendered LongEmbed docs).
* Similarity is inner-product over L2-normalized vectors (cosine).
* Transformers compatibility: the model config was saved with transformers 4.41.2.
  There is a known upstream issue where LlavaNextProcessor may break for >=4.52
  because `patch_size` becomes None. We apply a tiny *conditional* patch only if
  the attribute is missing, to keep the adapter usable in newer envs.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional

import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from .base import ModelAdapter
from ..records import QueryRecord, DocRecord
from ..utils import TokenStats, chunked
from ..index.faiss_flatip_cache import get_or_build_flatip, sha1_pairs


LLAMA3_TEMPLATE = (
    "<|start_header_id|>user<|end_header_id|>\n\n{}"
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n"
)


def _open_rgb(path: Optional[str]) -> Image.Image:
    if not path:
        return Image.new("RGB", (32, 32), color="white")
    return Image.open(path).convert("RGB")


class E5VAdapter(ModelAdapter):
    def __init__(
        self,
        model_name: str = "royokong/e5-v",
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        qry_bs: int = 16,
        doc_bs: int = 8,
        # prompts (keep defaults exactly as HF)
        llama3_template: str = LLAMA3_TEMPLATE,
        img_prompt_text: str = "<image>\nSummary above image in one word: ",
        text_prompt_text: str = "<sent>\nSummary above sentence in one word: ",
        # ---- faiss (FlatIP) index ----
        use_faiss: bool = True,
        faiss_cache_dir: Optional[str] = None,
        faiss_rebuild: bool = False,
        faiss_check_docids: bool = True,
        faiss_use_gpu: bool = False,
        faiss_gpu_device: int = 0,
        faiss_normalize: bool = True,
        # ---- transformers compatibility patch (optional) ----
        force_patch_size: Optional[int] = None,
    ):
        self.model_name = model_name
        self.dtype = dtype
        self.device = device
        self.qry_bs = int(qry_bs)
        self.doc_bs = int(doc_bs)
        self.use_faiss = bool(use_faiss)
        self.faiss_cache_dir = faiss_cache_dir or os.environ.get(
            "RETRIEVAL_WHEELS_FAISS_CACHE",
            os.path.join(os.path.expanduser("~"), ".cache", "retrieval_wheels", "faiss"),
        )
        self.faiss_rebuild = bool(faiss_rebuild)
        self.faiss_check_docids = bool(faiss_check_docids)
        self.faiss_use_gpu = bool(faiss_use_gpu)
        self.faiss_gpu_device = int(faiss_gpu_device)
        self.faiss_normalize = bool(faiss_normalize)

        # Keep prompts exactly like HF example.
        self.img_prompt = llama3_template.format(img_prompt_text)
        self.text_prompt = llama3_template.format(text_prompt_text)

        # Lazy import: only require transformers if this adapter is used.
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

        self.processor = LlavaNextProcessor.from_pretrained(self.model_name)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

        # ---- conditional compatibility patch for newer transformers ----
        # Some transformer versions expect LlavaNextProcessor.patch_size to be set.
        # The model's config has vision_config.patch_size (typically 14).
        try:
            patch = force_patch_size
            if patch is None:
                vc = getattr(getattr(self.model, "config", None), "vision_config", None)
                patch = getattr(vc, "patch_size", None)
            if patch is not None:
                if getattr(self.processor, "patch_size", None) is None:
                    setattr(self.processor, "patch_size", int(patch))
                ip = getattr(self.processor, "image_processor", None)
                if ip is not None and getattr(ip, "patch_size", None) is None:
                    setattr(ip, "patch_size", int(patch))
        except Exception:
            # Keep silent: if transformers already works, no need to touch.
            pass

        self._last_token_stats: Dict = {}

    @property
    def name(self) -> str:
        return "e5_v"

    @property
    def requires_doc_images(self) -> bool:
        return True

    def get_last_token_stats(self) -> dict:
        return getattr(self, "_last_token_stats", {})

    def _corpus_sig(self, corpus: List[DocRecord]) -> str:
        return sha1_pairs((d.docid, d.image_path or "") for d in corpus)

    @torch.no_grad()
    def _encode_text_batch(self, texts: List[str]) -> torch.Tensor:
        # HF example: processor([...], return_tensors="pt", padding=True).to('cuda')
        # NOTE: use explicit keyword `text=` to avoid positional-arg ambiguity
        # across transformers versions.
        inputs = self.processor(
            text=[self.text_prompt.replace("<sent>", t) for t in texts],
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        out = self.model(**inputs, output_hidden_states=True, return_dict=True)
        emb = out.hidden_states[-1][:, -1, :]
        return F.normalize(emb, dim=-1)

    @torch.no_grad()
    def _encode_image_batch(self, images: List[Image.Image]) -> torch.Tensor:
        # HF example: processor([img_prompt]*len(images), images, return_tensors="pt", padding=True).to('cuda')
        # NOTE: use explicit keywords to avoid treating text prompts as images.
        inputs = self.processor(
            text=[self.img_prompt] * len(images),
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        out = self.model(**inputs, output_hidden_states=True, return_dict=True)
        emb = out.hidden_states[-1][:, -1, :]
        return F.normalize(emb, dim=-1)

    @torch.no_grad()
    def _iter_doc_vecs_f32(self, corpus: List[DocRecord]) -> Iterator[np.ndarray]:
        for batch in tqdm(list(chunked(corpus, self.doc_bs)), desc="encode corpus images (E5-V)", unit="batch"):
            imgs = [_open_rgb(d.image_path) for d in batch]

            # token stats for multimodal inputs
            inputs = self.processor(
                text=[self.img_prompt] * len(imgs),
                images=imgs,
                return_tensors="pt",
                padding=True,
            )
            if "attention_mask" in inputs:
                lens = inputs["attention_mask"].sum(dim=1).tolist()
                self._d_img_stats.add_many([int(x) for x in lens])

            inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            out = self.model(**inputs, output_hidden_states=True, return_dict=True)
            emb = out.hidden_states[-1][:, -1, :]
            emb = F.normalize(emb, dim=-1) if self.faiss_normalize else emb
            xb = emb.detach().float().cpu().numpy().astype(np.float32, copy=False)
            yield np.ascontiguousarray(xb)

    def search(self, queries: List[QueryRecord], corpus: List[DocRecord], topk: int) -> Dict[str, Dict[str, float]]:
        self.validate_inputs(queries, corpus)

        # reset stats per run
        self._q_text_stats = TokenStats()
        self._d_img_stats = TokenStats()

        doc_ids = [d.docid for d in corpus]
        qids = [q.qid for q in queries]
        qtexts = [q.text or "" for q in queries]

        # token stats (queries) – match HF flow: build inputs via processor
        for batch in tqdm(list(chunked(qtexts, self.qry_bs)), desc="tokenize queries (E5-V)", unit="batch"):
            inputs = self.processor(
                text=[self.text_prompt.replace("<sent>", t) for t in list(batch)],
                return_tensors="pt",
                padding=True,
            )
            if "attention_mask" in inputs:
                lens = inputs["attention_mask"].sum(dim=1).tolist()
                self._q_text_stats.add_many([int(x) for x in lens])

        if not self.use_faiss:
            # brute-force (small corpora only)
            doc_reps = []
            for batch in tqdm(list(chunked(corpus, self.doc_bs)), desc="encode corpus images (E5-V)", unit="batch"):
                imgs = [_open_rgb(d.image_path) for d in batch]
                doc_reps.append(self._encode_image_batch(imgs).detach().cpu())
            d_reps = torch.cat(doc_reps, dim=0)  # (nd, D)

            q_reps = []
            for batch in tqdm(list(chunked(qtexts, self.qry_bs)), desc="encode queries (E5-V)", unit="batch"):
                q_reps.append(self._encode_text_batch(list(batch)).detach().cpu())
            q_reps = torch.cat(q_reps, dim=0)  # (nq, D)

            scores = q_reps @ d_reps.t()  # (nq, nd)
            run: Dict[str, Dict[str, float]] = {}
            for i in tqdm(range(scores.size(0)), desc="scoring+topk (E5-V)", unit="q"):
                row = scores[i]
                k = min(int(topk), row.numel())
                vals, idx = torch.topk(row, k=k, largest=True)
                run[qids[i]] = {doc_ids[j]: float(vals[t].item()) for t, j in enumerate(idx.tolist())}
        else:
            corpus_sig = self._corpus_sig(corpus)
            cfg = {
                "adapter": self.name,
                "model_name": self.model_name,
                "dtype": str(self.dtype),
                "device": str(self.device),
                "img_prompt": "Summary above image in one word",
                "text_prompt": "Summary above sentence in one word",
                "normalize": bool(self.faiss_normalize),
                "corpus_sig": corpus_sig,
                "index": "IndexFlatIP",
                "vec": "last_hidden_state[-1][-1]",
            }

            extra_meta = lambda: {
                "doc_seq_stats_raw": self._d_img_stats.to_raw(),
            }

            index, cached_docids, meta, loaded = get_or_build_flatip(
                cache_dir=self.faiss_cache_dir,
                cfg=cfg,
                doc_ids=doc_ids,
                encode_doc_batches=lambda: self._iter_doc_vecs_f32(corpus),
                rebuild=self.faiss_rebuild,
                check_docids=self.faiss_check_docids,
                use_gpu=self.faiss_use_gpu,
                gpu_device=self.faiss_gpu_device,
                normalize=False,  # we normalize ourselves if requested
                extra_meta=extra_meta,
            )
            if loaded and meta:
                self._d_img_stats = TokenStats.from_raw(meta.get("doc_seq_stats_raw", {}))

            # encode queries to float32 numpy
            q_batches: List[np.ndarray] = []
            for batch in tqdm(list(chunked(qtexts, self.qry_bs)), desc="encode queries (E5-V)", unit="batch"):
                emb = self._encode_text_batch(list(batch))
                emb = emb.detach().float().cpu().numpy().astype(np.float32, copy=False)
                q_batches.append(np.ascontiguousarray(emb))
            qv = np.concatenate(q_batches, axis=0)

            D, I = index.search(np.ascontiguousarray(qv), int(topk))
            run = {}
            for qi, qid in enumerate(qids):
                scores = D[qi].tolist()
                idxs = I[qi].tolist()
                dd = {}
                for s, j in zip(scores, idxs):
                    if int(j) < 0:
                        continue
                    dd[str(cached_docids[int(j)])] = float(s)
                run[str(qid)] = dd

        self._last_token_stats = {
            "text_side": self._q_text_stats.as_dict(),
            "image_side": self._d_img_stats.as_dict(),
            "breakdown": {
                "text_query_total_seq": self._q_text_stats.as_dict(),
                "image_doc_total_seq": self._d_img_stats.as_dict(),
            },
        }
        return run
