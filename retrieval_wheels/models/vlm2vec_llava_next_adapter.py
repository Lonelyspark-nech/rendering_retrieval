# -*- coding: utf-8 -*-
"""VLM2Vec (MMEB) adapter for TIGER-Lab/VLM2Vec-LLaVa-Next.

This adapter is **independent** from the existing Qwen2-VL-based VLM2Vec adapter.

It strictly follows the Hugging Face example for this model:
  - build ModelArguments(model_name=..., pooling='last', normalize=True, model_backbone='llava_next')
  - processor = load_processor(model_args)
  - model = MMEBModel.load(model_args).to('cuda', dtype=bf16).eval()
  - multimodal (image+text prompt) -> model(qry=...)['qry_reps']
  - text-only -> model(tgt=...)['tgt_reps']

Retrieval strategy (same wheel pattern):
  - build/load a FlatIP Faiss index on doc reps (qry branch)
  - encode query texts as tgt reps
  - search topk with Faiss
"""

from __future__ import annotations

from typing import Dict, List, Optional, Iterator

import os

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from .base import ModelAdapter
from ..records import QueryRecord, DocRecord
from ..utils import chunked, TokenStats
from ..index.faiss_flatip_cache import get_or_build_flatip, sha1_pairs


def _move_inputs_to_device(inputs: dict, device: str, dtype: torch.dtype) -> dict:
    """Move processor outputs to device.

    - pixel_values: move + cast to dtype
    - other tensors: move only
    """
    out = {}
    for k, v in inputs.items():
        if not isinstance(v, torch.Tensor):
            out[k] = v
            continue
        if k == "pixel_values" and torch.is_floating_point(v):
            out[k] = v.to(device=device, dtype=dtype)
        else:
            out[k] = v.to(device=device)
    return out


class VLM2VecLlavaNextText2ImageAdapter(ModelAdapter):
    """Text query (tgt) retrieval over rendered document images (qry) using VLM2Vec-LLaVa-Next."""

    def __init__(
        self,
        model_name: str = "TIGER-Lab/VLM2Vec-LLaVa-Next",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        # Keep the prompt format aligned with the HF example.
        doc_image_question: str = "What is in the image",
        qry_bs: int = 16,
        doc_bs: int = 8,
        # ---- faiss (FlatIP) index ----
        use_faiss: bool = True,
        faiss_cache_dir: Optional[str] = None,
        faiss_rebuild: bool = False,
        faiss_check_docids: bool = True,
        faiss_use_gpu: bool = False,
        faiss_gpu_device: int = 0,
        # If True, L2-normalize vectors before adding/searching (harmless when already normalized).
        faiss_normalize: bool = False,
    ):
        self.device = device
        self.dtype = dtype
        self.model_name = model_name

        self.doc_image_question = doc_image_question
        self.qry_bs = int(qry_bs)
        self.doc_bs = int(doc_bs)

        # faiss config
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

        # Import from the VLM2Vec repo (must be on PYTHONPATH)
        from src.model import MMEBModel
        from src.arguments import ModelArguments

        # load_processor is located at different places across VLM2Vec versions
        try:
            from src.utils import load_processor  # HF example
        except Exception:
            from src.model_utils import load_processor

        self.model_args = ModelArguments(
            model_name=self.model_name,
            pooling="last",
            normalize=True,
            model_backbone="llava_next",
        )
        self.processor = load_processor(self.model_args)
        self.model = MMEBModel.load(self.model_args).to(self.device, dtype=self.dtype).eval()

        self._last_token_stats = {}

    @property
    def name(self) -> str:
        return "vlm2vec_llava_next_text2image"

    @property
    def requires_doc_images(self) -> bool:
        return True

    def get_last_token_stats(self) -> dict:
        return getattr(self, "_last_token_stats", {})

    def _corpus_sig(self, corpus: List[DocRecord]) -> str:
        return sha1_pairs((d.docid, d.image_path or "") for d in corpus)

    def _doc_prompt(self) -> str:
        # HF example:
        #   '<image> Represent the given image with the following question: What is in the image'
        q = (self.doc_image_question or "").strip()
        return f"<image> Represent the given image with the following question: {q}".strip()

    @torch.no_grad()
    def _iter_doc_vecs_f32(self, docs: List[DocRecord]) -> Iterator[np.ndarray]:
        """Stream doc vectors as float32 numpy batches (for Faiss add)."""
        prompt = self._doc_prompt()
        for batch in tqdm(list(chunked(docs, self.doc_bs)), desc="encode corpus images (VLM2Vec-LLaVa-Next)", unit="batch"):
            imgs = []
            texts = []
            for d in batch:
                if not d.image_path:
                    imgs.append(Image.new("RGB", (32, 32), (255, 255, 255)))
                else:
                    imgs.append(Image.open(d.image_path).convert("RGB"))
                texts.append(prompt)

            inputs = self.processor(text=texts, images=imgs, return_tensors="pt", padding=True)
            if "attention_mask" in inputs:
                lens = inputs["attention_mask"].sum(dim=1).tolist()
                self._d_stats.add_many(lens)

            inputs = _move_inputs_to_device(inputs, self.device, self.dtype)
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                out = self.model(qry=inputs)["qry_reps"]  # (B, D)
            xb = out.detach().float().cpu().numpy().astype(np.float32, copy=False)
            if self.faiss_normalize:
                from ..index.faiss_flatip_cache import l2_normalize_inplace
                l2_normalize_inplace(xb)
            yield np.ascontiguousarray(xb)

    @torch.no_grad()
    def _encode_query_texts_as_tgt(self, queries: List[QueryRecord]) -> torch.Tensor:
        reps = []
        texts = [q.text or "" for q in queries]
        for batch in tqdm(list(chunked(texts, self.qry_bs)), desc="encode queries text (VLM2Vec-LLaVa-Next)", unit="batch"):
            inputs = self.processor(text=list(batch), images=None, return_tensors="pt", padding=True)
            if "attention_mask" in inputs:
                lens = inputs["attention_mask"].sum(dim=1).tolist()
                self._q_stats.add_many(lens)

            inputs = _move_inputs_to_device(inputs, self.device, self.dtype)
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                out = self.model(tgt=inputs)["tgt_reps"]  # (B, D)
            reps.append(out)
        return torch.cat(reps, dim=0)

    def search(self, queries: List[QueryRecord], corpus: List[DocRecord], topk: int) -> Dict[str, Dict[str, float]]:
        # stats per run
        self._q_stats = TokenStats()
        self._d_stats = TokenStats()

        doc_ids = [d.docid for d in corpus]
        qids = [q.qid for q in queries]

        if not self.use_faiss:
            # brute-force (small corpora only)
            # Encode docs
            doc_reps = []
            for xb in self._iter_doc_vecs_f32(corpus):
                doc_reps.append(torch.from_numpy(xb))
            doc_reps = torch.cat(doc_reps, dim=0).to(self.device)

            qry_reps = self._encode_query_texts_as_tgt(queries)
            with torch.no_grad():
                sim = self.model.compute_similarity(doc_reps, qry_reps)  # (nd, nq)
            scores = sim.transpose(0, 1).float().cpu()  # (nq, nd)
            run: Dict[str, Dict[str, float]] = {}
            for i in tqdm(range(scores.size(0)), desc="scoring+topk (VLM2Vec-LLaVa-Next)", unit="q"):
                row = scores[i]
                k = min(int(topk), row.numel())
                vals, idx = torch.topk(row, k=k, largest=True)
                run[str(qids[i])] = {str(doc_ids[j]): float(vals[t].item()) for t, j in enumerate(idx.tolist())}
        else:
            corpus_sig = self._corpus_sig(corpus)
            cfg = {
                "adapter": self.name,
                "model_name": self.model_name,
                "doc_image_question": self.doc_image_question,
                "dtype": str(self.dtype),
                "corpus_sig": corpus_sig,
                "index": "IndexFlatIP",
                "vec": "doc=qry_reps, query=tgt_reps",
            }

            def _encode_batches() -> Iterator[np.ndarray]:
                return self._iter_doc_vecs_f32(corpus)

            extra_meta = lambda: {"doc_stats_raw": self._d_stats.to_raw()}
            index, cached_docids, meta, loaded = get_or_build_flatip(
                cache_dir=self.faiss_cache_dir,
                cfg=cfg,
                doc_ids=doc_ids,
                encode_doc_batches=_encode_batches,
                rebuild=self.faiss_rebuild,
                check_docids=self.faiss_check_docids,
                use_gpu=self.faiss_use_gpu,
                gpu_device=self.faiss_gpu_device,
                normalize=False,  # model outputs normalized when normalize=True
                extra_meta=extra_meta,
            )

            if loaded and meta:
                self._d_stats = TokenStats.from_raw(meta.get("doc_stats_raw", {}))

            qry_reps = self._encode_query_texts_as_tgt(queries)
            qv = qry_reps.detach().float().cpu().numpy().astype(np.float32, copy=False)
            if self.faiss_normalize:
                from ..index.faiss_flatip_cache import l2_normalize_inplace
                l2_normalize_inplace(qv)
            qv = np.ascontiguousarray(qv)

            D, I = index.search(qv, int(topk))
            run = {}
            for i, qid in enumerate(qids):
                dd = {}
                for j, s in zip(I[i].tolist(), D[i].tolist()):
                    if j < 0:
                        continue
                    dd[str(cached_docids[j])] = float(s)
                run[str(qid)] = dd

        self._last_token_stats = {
            "text_side": self._q_stats.as_dict(),
            "image_side": self._d_stats.as_dict(),
            "breakdown": {
                "text_query": self._q_stats.as_dict(),
                "image_doc_total_seq": self._d_stats.as_dict(),
            },
        }
        return run
