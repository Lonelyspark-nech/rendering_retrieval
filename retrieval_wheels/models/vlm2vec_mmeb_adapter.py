# -*- coding: utf-8 -*-
from typing import Dict, List, Optional, Iterator
from tqdm.auto import tqdm
from PIL import Image
import torch
import numpy as np
import os

from .base import ModelAdapter
from ..records import QueryRecord, DocRecord
from ..utils import chunked, TokenStats
from ..index.faiss_flatip_cache import get_or_build_flatip, sha1_pairs

from src.model import MMEBModel
from src.arguments import ModelArguments
from src.model_utils import load_processor, QWEN2_VL, vlm_image_tokens


def _ensure_grid_thw_b13(g: torch.Tensor) -> torch.Tensor:
    """
    Normalize image_grid_thw to shape (B, 1, 3).
    Accepts:
      - (B, 3) -> (B, 1, 3)
      - (3,)   -> (1, 1, 3)
      - (B, 1, 3) stays
    """
    if g.dim() == 1 and g.numel() == 3:
        return g.view(1, 1, 3)
    if g.dim() == 2 and g.size(-1) == 3:
        return g.unsqueeze(1)
    return g


def _pack_qwen2vl_vision_inputs(inputs: dict) -> dict:
    """
    Make Qwen2-VL vision inputs compatible with VLM2Vec backbone.

    Expected (safe) format for this VLM2Vec/Qwen2-VL implementation:
      - image_grid_thw: (B, 1, 3)
      - pixel_values:  (B, 1, P, D)
        where P = t*h*w for that image.

    Processor (slow/fast) may output:
      - pixel_values: (sum_P, D)   (packed across batch)
      - image_grid_thw: (B, 3)

    We convert packed (sum_P, D) -> (B, 1, P, D) using grid_thw.
    """
    if "image_grid_thw" not in inputs or "pixel_values" not in inputs:
        return inputs
    g = inputs["image_grid_thw"]
    pv = inputs["pixel_values"]
    if not isinstance(g, torch.Tensor) or not isinstance(pv, torch.Tensor):
        return inputs

    g = _ensure_grid_thw_b13(g)  # (B,1,3)
    inputs["image_grid_thw"] = g

    # If pixel_values already has batch/img dims, normalize lightly
    # Cases:
    #   pv: (B, P, D)     -> (B,1,P,D)
    #   pv: (B,1,P,D)     -> ok
    #   pv: (sum_P, D)    -> pack to (B,1,P,D)
    if pv.dim() == 4:
        # (B,1,P,D) or (B,N,P,D) -> keep
        return inputs

    B, N, _ = g.shape  # N should be 1 in your usage
    counts = (g[..., 0] * g[..., 1] * g[..., 2]).tolist()  # (B,N) python list
    flat_counts = [int(c) for row in counts for c in row]  # length B*N
    total = sum(flat_counts)

    if pv.dim() == 3:
        # (B,P,D) or (B*N,P,D)
        if pv.size(0) == B and N == 1:
            inputs["pixel_values"] = pv.unsqueeze(1)  # (B,1,P,D)
            return inputs
        if pv.size(0) == B * N:
            inputs["pixel_values"] = pv.view(B, N, pv.size(1), pv.size(2))
            return inputs
        # otherwise fall through (rare)

    if pv.dim() == 2:
        # packed: (sum_P, D)
        if pv.size(0) != total:
            # mismatch: don't guess; leave as-is and let model throw with clearer context
            return inputs

        D = pv.size(1)
        # split by each image count
        chunks = torch.split(pv, flat_counts, dim=0)  # list length B*N

        if len(set(flat_counts)) == 1:
            P = flat_counts[0]
            packed = pv.view(B * N, P, D).view(B, N, P, D)  # (B,N,P,D)
            inputs["pixel_values"] = packed
            return inputs

        # variable P: pad to maxP (unlikely in your fixed rendering settings)
        maxP = max(flat_counts)
        padded = []
        for x, c in zip(chunks, flat_counts):
            if c < maxP:
                pad = x.new_zeros((maxP - c, D))
                x = torch.cat([x, pad], dim=0)
            padded.append(x)
        stacked = torch.stack(padded, dim=0).view(B, N, maxP, D)  # (B,N,maxP,D)
        inputs["pixel_values"] = stacked
        return inputs

    return inputs


def _move_inputs_to_device(inputs: dict, device: str, dtype: torch.dtype) -> dict:
    """
    - pixel_values: move + cast to dtype
    - other tensors: move only (keep dtype)
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


class VLM2VecMMEBText2ImageAdapter(ModelAdapter):
    """
    rendered corpus:
      - query: LongEmbed queries 的纯文本 -> tgt_reps
      - doc: rendered images/<docid>.png (+固定prompt + image token) -> qry_reps
      - sim: model.compute_similarity(qry, tgt)
    """
    def __init__(
        self,
        model_name: str = "/data3/sunbo/models/Qwen/Qwen2-VL-7B-Instruct",
        checkpoint_path: str = "/data3/sunbo/models/TIGER-Lab/VLM2Vec-Qwen2VL-7B",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        doc_image_prompt: str = "Represent the given image.",
        qry_bs: int = 16,
        doc_bs: int = 8,
        # ---- faiss (FlatIP) index ----
        use_faiss: bool = True,
        faiss_cache_dir: Optional[str] = None,
        faiss_rebuild: bool = False,
        faiss_check_docids: bool = True,
        faiss_use_gpu: bool = False,
        faiss_gpu_device: int = 0,
        faiss_normalize: bool = False,
    ):
        self.device = device
        self.dtype = dtype
        self.doc_image_prompt = doc_image_prompt
        self.qry_bs = int(qry_bs)
        self.doc_bs = int(doc_bs)

        self.model_name = model_name
        self.checkpoint_path = checkpoint_path

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

        self.model_args = ModelArguments(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            pooling="last",
            normalize=True,
            model_backbone="qwen2_vl",
            lora=True,
        )
        self.processor = load_processor(self.model_args)
        self.model = MMEBModel.load(self.model_args).to(self.device, dtype=self.dtype).eval()

        self.image_token = vlm_image_tokens[QWEN2_VL]
        self._last_token_stats = {}

    def _corpus_sig(self, corpus: List[DocRecord]) -> str:
        # signature based on docid + image_path (fast; assumes images are immutable unless rebuild=True)
        return sha1_pairs((d.docid, d.image_path or "") for d in corpus)

    @torch.no_grad()
    def _iter_doc_vecs_f32(self, docs: List[DocRecord]) -> Iterator[np.ndarray]:
        """Stream doc vectors as float32 numpy batches (for Faiss add)."""
        prompt = f"{self.image_token} {self.doc_image_prompt}".strip()
        for batch in tqdm(list(chunked(docs, self.doc_bs)), desc="encode corpus images (MMEB)", unit="batch"):
            imgs = []
            texts = []
            for d in batch:
                if not d.image_path:
                    imgs.append(Image.new("RGB", (32, 32), (255, 255, 255)))
                else:
                    imgs.append(Image.open(d.image_path).convert("RGB"))
                texts.append(prompt)

            inputs = self.processor(text=texts, images=imgs, return_tensors="pt", padding=True)

            # token length for multimodal input (text + image tokens)
            if "attention_mask" in inputs:
                lens = inputs["attention_mask"].sum(dim=1).tolist()
                self._d_img_stats.add_many(lens)

            # vision tokens (t*h*w) based on raw grid_thw if present
            if "image_grid_thw" in inputs and isinstance(inputs["image_grid_thw"], torch.Tensor):
                g0 = inputs["image_grid_thw"]
                if g0.dim() == 2 and g0.size(-1) == 3:
                    vision = (g0[:, 0] * g0[:, 1] * g0[:, 2]).tolist()
                    self._d_vision_stats.add_many(vision)

            inputs = _pack_qwen2vl_vision_inputs(inputs)
            inputs = _move_inputs_to_device(inputs, self.device, self.dtype)

            with torch.autocast(device_type="cuda", dtype=self.dtype):
                out = self.model(qry=inputs)["qry_reps"]  # (B, D)
            xb = out.detach().float().cpu().numpy().astype(np.float32, copy=False)
            if self.faiss_normalize:
                # harmless if already normalized
                from ..index.faiss_flatip_cache import l2_normalize_inplace
                l2_normalize_inplace(xb)
            yield xb

    @property
    def name(self) -> str:
        return "vlm2vec_mmeb_text2image"

    @property
    def requires_doc_images(self) -> bool:
        return True

    def get_last_token_stats(self) -> dict:
        return getattr(self, "_last_token_stats", {})

    @torch.no_grad()
    def _encode_doc_images_as_qry(self, docs: List[DocRecord]) -> torch.Tensor:
        reps = []
        prompt = f"{self.image_token} {self.doc_image_prompt}".strip()

        for batch in tqdm(list(chunked(docs, self.doc_bs)), desc="encode corpus images (MMEB)", unit="batch"):
            imgs = []
            texts = []
            for d in batch:
                if not d.image_path:
                    imgs.append(Image.new("RGB", (32, 32), (255, 255, 255)))
                else:
                    imgs.append(Image.open(d.image_path).convert("RGB"))
                texts.append(prompt)

            inputs = self.processor(text=texts, images=imgs, return_tensors="pt", padding=True)

            # token length for multimodal input (text + image tokens)
            if "attention_mask" in inputs:
                lens = inputs["attention_mask"].sum(dim=1).tolist()
                self._d_img_stats.add_many(lens)

            # vision tokens (t*h*w) based on raw grid_thw if present
            if "image_grid_thw" in inputs and isinstance(inputs["image_grid_thw"], torch.Tensor):
                g0 = inputs["image_grid_thw"]
                if g0.dim() == 2 and g0.size(-1) == 3:
                    vision = (g0[:, 0] * g0[:, 1] * g0[:, 2]).tolist()
                    self._d_vision_stats.add_many(vision)

            # ✅ 关键：把 (sum_P, D) pack 成 (B,1,P,D)，grid_thw 变成 (B,1,3)
            inputs = _pack_qwen2vl_vision_inputs(inputs)
            inputs = _move_inputs_to_device(inputs, self.device, self.dtype)

            # 如需确认形状，取消注释（只看第一 batch 一次即可）
            # print("pixel_values:", inputs["pixel_values"].shape)
            # print("image_grid_thw:", inputs["image_grid_thw"].shape)

            with torch.autocast(device_type="cuda", dtype=self.dtype):
                out = self.model(qry=inputs)["qry_reps"]  # (B, D)
            reps.append(out)

        return torch.cat(reps, dim=0)

    @torch.no_grad()
    def _encode_query_texts_as_tgt(self, queries: List[QueryRecord]) -> torch.Tensor:
        reps = []
        texts = [q.text or "" for q in queries]
        for batch in tqdm(list(chunked(texts, self.qry_bs)), desc="encode queries text (MMEB)", unit="batch"):
            inputs = self.processor(text=list(batch), images=None, return_tensors="pt", padding=True)

            if "attention_mask" in inputs:
                lens = inputs["attention_mask"].sum(dim=1).tolist()
                self._q_text_stats.add_many(lens)

            inputs = _move_inputs_to_device(inputs, self.device, self.dtype)
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                out = self.model(tgt=inputs)["tgt_reps"]  # (B, D)
            reps.append(out)

        return torch.cat(reps, dim=0)

    def search(self, queries: List[QueryRecord], corpus: List[DocRecord], topk: int) -> Dict[str, Dict[str, float]]:
        # reset stats per run
        self._q_text_stats = TokenStats()
        self._d_img_stats = TokenStats()
        self._d_vision_stats = TokenStats()

        doc_ids = [d.docid for d in corpus]
        qids = [q.qid for q in queries]

        # ---- build/load faiss index over doc reps (qry branch) ----
        if not self.use_faiss:
            # legacy brute-force (small corpora only)
            doc_reps = self._encode_doc_images_as_qry(corpus)     # (nd, D)
            qry_reps = self._encode_query_texts_as_tgt(queries)   # (nq, D)
            with torch.no_grad():
                sim = self.model.compute_similarity(doc_reps, qry_reps)  # (nd, nq)
            scores = sim.transpose(0, 1).float().cpu()  # (nq, nd)
            run: Dict[str, Dict[str, float]] = {}
            for i in tqdm(range(scores.size(0)), desc="scoring+topk (MMEB)", unit="q"):
                row = scores[i]
                k = min(int(topk), row.numel())
                vals, idx = torch.topk(row, k=k, largest=True)
                run[qids[i]] = {doc_ids[j]: float(vals[t].item()) for t, j in enumerate(idx.tolist())}
        else:
            corpus_sig = self._corpus_sig(corpus)
            cfg = {
                "adapter": self.name,
                "model_name": self.model_name,
                "checkpoint_path": self.checkpoint_path,
                "doc_image_prompt": self.doc_image_prompt,
                "dtype": str(self.dtype),
                "corpus_sig": corpus_sig,
                "index": "IndexFlatIP",
                "vec": "doc=qry_reps, query=tgt_reps",
            }

            def _encode_batches() -> Iterator[np.ndarray]:
                return self._iter_doc_vecs_f32(corpus)

            # build/load index; persist doc-side token stats for reuse
            extra_meta = lambda: {
                "doc_img_stats_raw": self._d_img_stats.to_raw(),
                "doc_vision_stats_raw": self._d_vision_stats.to_raw(),
            }
            index, cached_docids, meta, loaded = get_or_build_flatip(
                cache_dir=self.faiss_cache_dir,
                cfg=cfg,
                doc_ids=doc_ids,
                encode_doc_batches=_encode_batches,
                rebuild=self.faiss_rebuild,
                check_docids=self.faiss_check_docids,
                use_gpu=self.faiss_use_gpu,
                gpu_device=self.faiss_gpu_device,
                normalize=False,  # model already outputs normalized reps (normalize=True)
                extra_meta=extra_meta,
            )

            if loaded and meta:
                # restore doc-side stats without re-encoding
                self._d_img_stats = TokenStats.from_raw(meta.get("doc_img_stats_raw", {}))
                self._d_vision_stats = TokenStats.from_raw(meta.get("doc_vision_stats_raw", {}))

            # encode queries (tgt branch)
            qry_reps = self._encode_query_texts_as_tgt(queries)   # (nq, D)
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
            "text_side": self._q_text_stats.as_dict(),
            "image_side": self._d_img_stats.as_dict(),
            "breakdown": {
                "text_query": self._q_text_stats.as_dict(),
                "image_doc_total_seq": self._d_img_stats.as_dict(),
                "image_doc_vision_tokens": self._d_vision_stats.as_dict(),
            },
        }
        return run
