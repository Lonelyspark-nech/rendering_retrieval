# -*- coding: utf-8 -*-
"""ColPali adapter (ColBERT-style multi-vector late interaction).

Goals:
  - Follow the HF/ColPali Engine usage pattern strictly:
        batch_images  = processor.process_images(images).to(model.device)
        batch_queries = processor.process_queries(queries).to(model.device)
        image_embeddings = model(**batch_images)
        query_embeddings = model(**batch_queries)
        scores = processor.score_multi_vector(query_embeddings, image_embeddings)

  - Integrate with the existing retrieval_wheels runner by implementing ModelAdapter.search().

Efficiency:
  - Full-corpus late-interaction is too expensive. We do a two-stage search:
      1) Build a Faiss IndexFlatIP over pooled (mean) single-vector representations.
      2) For each query, preselect_k candidates from Faiss, then rerank with
         processor.score_multi_vector (late interaction).

This keeps the scoring stage faithful to ColPali while fitting the existing Faiss-based wheel.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

import os
import json
import hashlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from .base import ModelAdapter
from ..records import QueryRecord, DocRecord
from ..utils import TokenStats, chunked
from ..index.faiss_flatip_cache import get_or_build_flatip, sha1_pairs


def _hash_cfg(obj: Dict) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def _open_rgb(path: Optional[str]) -> Image.Image:
    if not path:
        return Image.new("RGB", (32, 32), color="white")
    return Image.open(path).convert("RGB")


def _as_tensor(x):
    """ColPali outputs are usually torch.Tensor, but keep this resilient."""
    if isinstance(x, torch.Tensor):
        return x
    if hasattr(x, "embeddings"):
        return x.embeddings
    raise TypeError(f"Unexpected embedding type: {type(x)}")


def _to_cache_dtype(x: torch.Tensor, dtype_tag: str) -> torch.Tensor:
    if dtype_tag == "bf16":
        return x.to(torch.bfloat16)
    if dtype_tag == "fp16":
        return x.to(torch.float16)
    if dtype_tag == "fp32":
        return x.to(torch.float32)
    raise ValueError(f"Unknown doc_multi_cache_dtype: {dtype_tag}")


class ColPaliAdapter(ModelAdapter):
    def __init__(
        self,
        model_name: str = "/data3/sunbo/models/vidore/colpali-v1.2",
        device_map: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        qry_bs: int = 8,
        doc_bs: int = 8,
        rerank_doc_bs: int = 8,
        preselect_k: int = 200,
        # ---- rerank acceleration ----
        # Caching doc multi-vector embeddings avoids re-encoding the same candidate images for every query.
        # This is critical for speed when #queries is large.
        cache_doc_multi: str = "memory",  # "none" | "memory" | "disk"
        doc_multi_cache_dtype: str = "bf16",  # "bf16" | "fp16" | "fp32"
        # ---- faiss (FlatIP) index ----
        use_faiss: bool = True,
        faiss_cache_dir: Optional[str] = None,
        faiss_rebuild: bool = False,
        faiss_check_docids: bool = True,
        faiss_use_gpu: bool = False,
        faiss_gpu_device: int = 0,
        faiss_normalize: bool = True,
    ):
        """Create a ColPali adapter.

        Args:
            model_name: HF repo id or local path (you use local).
            device_map: e.g., "cuda:0" (passed to from_pretrained).
            dtype: torch dtype for weights.
            qry_bs/doc_bs: encoding batch sizes.
            rerank_doc_bs: batch size for encoding candidate docs during rerank.
            preselect_k: number of Faiss candidates to rerank per query (>= topk).
            faiss_*: see other adapters; cache_dir defaults under ~/.cache.
        """
        self.model_name = model_name
        self.device_map = device_map
        self.dtype = dtype
        self.qry_bs = int(qry_bs)
        self.doc_bs = int(doc_bs)
        self.rerank_doc_bs = int(rerank_doc_bs)
        self.preselect_k = int(preselect_k)

        self.cache_doc_multi = str(cache_doc_multi)
        if self.cache_doc_multi not in {"none", "memory", "disk"}:
            raise ValueError("cache_doc_multi must be one of: none|memory|disk")
        self.doc_multi_cache_dtype = str(doc_multi_cache_dtype)
        if self.doc_multi_cache_dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError("doc_multi_cache_dtype must be one of: bf16|fp16|fp32")

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

        # Lazy import so the wheel can run in envs without colpali-engine unless this adapter is used.
        from colpali_engine.models import ColPali, ColPaliProcessor

        self.model = ColPali.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=self.device_map,
        ).eval()
        self.processor = ColPaliProcessor.from_pretrained(self.model_name)

        self._last_token_stats: Dict = {}

    @property
    def name(self) -> str:
        return "colpali"

    @property
    def requires_doc_images(self) -> bool:
        return True

    def get_last_token_stats(self) -> dict:
        return getattr(self, "_last_token_stats", {})

    def _corpus_sig(self, corpus: List[DocRecord]) -> str:
        # signature based on docid + image_path (fast; assumes images are immutable unless rebuild=True)
        return sha1_pairs((d.docid, d.image_path or "") for d in corpus)

    def _docmulti_cache_path(self, corpus_sig: str) -> str:
        cfg = {
            "adapter": self.name,
            "model_name": self.model_name,
            "dtype": str(self.dtype),
            "corpus_sig": corpus_sig,
            "doc_multi_cache_dtype": self.doc_multi_cache_dtype,
        }
        key = _hash_cfg(cfg)
        os.makedirs(self.faiss_cache_dir, exist_ok=True)
        return os.path.join(self.faiss_cache_dir, f"docmulti_{key}.pt")

    @torch.no_grad()
    def _encode_corpus_multi_to_cpu(self, corpus: List[DocRecord], doc_seq_stats: TokenStats) -> torch.Tensor:
        """Encode full corpus images once and keep doc multi-vector embeddings on CPU.

        Returns: (N, Ld, D) CPU tensor.
        """
        batches: List[torch.Tensor] = []
        expected_L: Optional[int] = None

        for batch in tqdm(list(chunked(corpus, self.doc_bs)), desc="encode corpus images (ColPali, cache multi)", unit="batch"):
            embs_gpu = self._encode_images_multi(list(batch))  # (B, Ld, D) on GPU
            lens = [int(embs_gpu.size(1))] * int(embs_gpu.size(0))
            doc_seq_stats.add_many(lens)

            if expected_L is None:
                expected_L = int(embs_gpu.size(1))
            elif int(embs_gpu.size(1)) != expected_L:
                raise ValueError(
                    f"ColPali produced variable doc embedding length (got {int(embs_gpu.size(1))} vs expected {expected_L}). "
                    "This adapter assumes fixed-length multi-vectors for fast caching. "
                    "Set cache_doc_multi='none' to fall back to on-the-fly encoding."
                )

            embs_cpu = _to_cache_dtype(embs_gpu, self.doc_multi_cache_dtype).detach().to("cpu").contiguous()
            batches.append(embs_cpu)

        if not batches:
            raise ValueError("Empty corpus: no document images to encode.")

        return torch.cat(batches, dim=0).contiguous()

    @torch.no_grad()
    def _encode_images_multi(self, docs: List[DocRecord]) -> torch.Tensor:
        """Encode doc images -> multi-vector embeddings (B, Ld, D) on GPU."""
        imgs = [_open_rgb(d.image_path) for d in docs]
        batch_images = self.processor.process_images(imgs).to(self.model.device)
        image_embeddings = self.model(**batch_images)
        return _as_tensor(image_embeddings)

    @torch.no_grad()
    def _encode_queries_multi(self, queries: List[QueryRecord]) -> torch.Tensor:
        texts = [q.text or "" for q in queries]
        batch_queries = self.processor.process_queries(texts).to(self.model.device)
        query_embeddings = self.model(**batch_queries)
        return _as_tensor(query_embeddings)

    def _pool_mean(self, embs: torch.Tensor) -> torch.Tensor:
        """Mean pool multi-vector -> single vector (B, D)."""
        pooled = embs.mean(dim=1)
        if self.faiss_normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled

    @torch.no_grad()
    def _iter_doc_vecs_f32(self, corpus: List[DocRecord], doc_seq_stats: TokenStats) -> Iterator[np.ndarray]:
        """Stream pooled doc vectors as float32 numpy batches (for Faiss add)."""
        for batch in tqdm(list(chunked(corpus, self.doc_bs)), desc="encode corpus images (ColPali)", unit="batch"):
            embs = self._encode_images_multi(list(batch))  # (B, Ld, D)
            # token/patch length stats
            lens = [int(embs.size(1))] * int(embs.size(0))
            doc_seq_stats.add_many(lens)

            pooled = self._pool_mean(embs).detach().float().cpu().numpy().astype(np.float32, copy=False)
            yield np.ascontiguousarray(pooled)

    def search(self, queries: List[QueryRecord], corpus: List[DocRecord], topk: int) -> Dict[str, Dict[str, float]]:
        self.validate_inputs(queries, corpus)

        # reset stats per run
        q_seq_stats = TokenStats()
        d_seq_stats = TokenStats()
        d_seq_stats_rerank = TokenStats()

        doc_ids = [d.docid for d in corpus]
        qids = [q.qid for q in queries]

        if not self.use_faiss:
            raise ValueError("ColPaliAdapter currently requires Faiss preselection (use_faiss=True).")

        # ---- build/load faiss index over pooled doc reps ----
        corpus_sig = self._corpus_sig(corpus)

        # ---- optional: cache corpus multi-vector embeddings to accelerate rerank ----
        doc_multi_cpu: Optional[torch.Tensor] = None
        if self.cache_doc_multi != "none":
            docmulti_path = self._docmulti_cache_path(corpus_sig)
            if self.cache_doc_multi == "disk" and (not self.faiss_rebuild) and os.path.exists(docmulti_path):
                payload = torch.load(docmulti_path, map_location="cpu")
                try:
                    if payload.get("docids") == doc_ids and "emb" in payload:
                        doc_multi_cpu = payload["emb"]
                    else:
                        doc_multi_cpu = None
                except Exception:
                    doc_multi_cpu = None

            if doc_multi_cpu is None:
                # Encode once and keep on CPU; this avoids re-encoding candidate docs for every query.
                doc_multi_seq_stats = TokenStats()
                doc_multi_cpu = self._encode_corpus_multi_to_cpu(corpus, doc_multi_seq_stats)
                # If we didn't have doc stats from a cached faiss meta, use the encoded stats.
                d_seq_stats = doc_multi_seq_stats
                if self.cache_doc_multi == "disk":
                    torch.save(
                        {"docids": doc_ids, "emb": doc_multi_cpu, "dtype_tag": self.doc_multi_cache_dtype},
                        docmulti_path,
                    )
        cfg = {
            "adapter": self.name,
            "model_name": self.model_name,
            "dtype": str(self.dtype),
            "corpus_sig": corpus_sig,
            "index": "IndexFlatIP",
            "vec": "mean_pool(image_embeddings)",
            "normalize": bool(self.faiss_normalize),
        }

        def _encode_batches() -> Iterator[np.ndarray]:
            # If we already encoded doc multi-vectors (for rerank acceleration), reuse them to build the pooled index.
            if doc_multi_cpu is not None:
                for start in range(0, doc_multi_cpu.size(0), self.doc_bs):
                    embs = doc_multi_cpu[start : start + self.doc_bs]
                    pooled = self._pool_mean(embs).detach().float().cpu().numpy().astype(np.float32, copy=False)
                    yield np.ascontiguousarray(pooled)
            else:
                yield from self._iter_doc_vecs_f32(corpus, d_seq_stats)

        extra_meta = lambda: {
            "doc_seq_stats_raw": d_seq_stats.to_raw(),
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
            normalize=False,  # we normalize ourselves in _pool_mean if requested
            extra_meta=extra_meta,
        )

        if loaded and meta:
            d_seq_stats = TokenStats.from_raw(meta.get("doc_seq_stats_raw", {}))

        # ---- encode queries (multi + pooled for faiss) ----
        # We'll keep pooled vectors on CPU for Faiss search; multi-vectors moved per query for rerank.
        q_pooled_all: List[np.ndarray] = []
        q_multi_all: List[torch.Tensor] = []

        for batch in tqdm(list(chunked(queries, self.qry_bs)), desc="encode queries (ColPali)", unit="batch"):
            q_multi = self._encode_queries_multi(list(batch))  # (B, Lq, D) on GPU
            lens = [int(q_multi.size(1))] * int(q_multi.size(0))
            q_seq_stats.add_many(lens)

            q_pooled = self._pool_mean(q_multi).detach().float().cpu().numpy().astype(np.float32, copy=False)
            q_pooled_all.append(np.ascontiguousarray(q_pooled))

            # store multi on CPU (bf16/fp16) to save GPU memory
            q_multi_all.extend([q_multi[i].detach().to("cpu") for i in range(q_multi.size(0))])

        qv = np.concatenate(q_pooled_all, axis=0)

        pre_k = max(int(topk), int(self.preselect_k))
        D, I = index.search(np.ascontiguousarray(qv), int(pre_k))

        # ---- rerank candidates with exact late-interaction ----
        run: Dict[str, Dict[str, float]] = {}

        if doc_multi_cpu is not None:
            # Fast path: use cached doc multi-vectors (no per-query image encoding).
            if int(doc_multi_cpu.size(0)) != int(len(cached_docids)):
                raise ValueError(
                    f"Doc multi cache size mismatch: doc_multi_cpu has {int(doc_multi_cpu.size(0))} docs, "
                    f"but faiss index has {int(len(cached_docids))} docids."
                )

            for qi, qid in tqdm(list(enumerate(qids)), desc="rerank (ColPali)", unit="q"):
                cand_idx = [int(j) for j in I[qi].tolist() if int(j) >= 0]
                if not cand_idx:
                    run[str(qid)] = {}
                    continue

                q_multi = q_multi_all[qi].unsqueeze(0).to(self.model.device)  # (1, Lq, D)
                all_docids: List[str] = []
                all_scores: List[float] = []

                for idxs in chunked(cand_idx, self.rerank_doc_bs):
                    idxs = [int(x) for x in idxs if int(x) >= 0]
                    if not idxs:
                        continue
                    img_embs = doc_multi_cpu[idxs].to(self.model.device, non_blocking=False)  # (B, Ld, D)
                    d_seq_stats_rerank.add_many([int(img_embs.size(1))] * int(img_embs.size(0)))

                    s = self.processor.score_multi_vector(q_multi, img_embs)  # (1, B)
                    s = _as_tensor(s).squeeze(0).detach().float().cpu().tolist()

                    all_docids.extend([str(cached_docids[j]) for j in idxs])
                    all_scores.extend([float(x) for x in s])

                # pick topk
                arr = np.asarray(all_scores, dtype=np.float32)
                k = min(int(topk), int(arr.size))
                if k <= 0:
                    run[str(qid)] = {}
                    continue
                top_pos = np.argpartition(-arr, kth=k - 1)[:k]
                top_pos = top_pos[np.argsort(-arr[top_pos], kind="mergesort")]
                run[str(qid)] = {str(all_docids[p]): float(arr[p]) for p in top_pos.tolist()}
        else:
            # Slow path: encode candidate docs on-the-fly.
            doc_by_id = {d.docid: d for d in corpus}

            for qi, qid in tqdm(list(enumerate(qids)), desc="rerank (ColPali)", unit="q"):
                cand_idx = [int(j) for j in I[qi].tolist() if int(j) >= 0]
                if not cand_idx:
                    run[str(qid)] = {}
                    continue

                cand_docs: List[DocRecord] = []
                for j in cand_idx:
                    if j < 0 or j >= len(cached_docids):
                        continue
                    did = str(cached_docids[j])
                    drec = doc_by_id.get(did)
                    if drec is not None:
                        cand_docs.append(drec)
                if not cand_docs:
                    run[str(qid)] = {}
                    continue

                q_multi = q_multi_all[qi].unsqueeze(0).to(self.model.device)
                all_docids: List[str] = []
                all_scores: List[float] = []

                for db in chunked(cand_docs, self.rerank_doc_bs):
                    img_embs = self._encode_images_multi(list(db))  # (B, Ld, D)
                    d_seq_stats_rerank.add_many([int(img_embs.size(1))] * int(img_embs.size(0)))
                    s = self.processor.score_multi_vector(q_multi, img_embs)  # (1, B)
                    s = _as_tensor(s).squeeze(0).detach().float().cpu().tolist()
                    all_docids.extend([d.docid for d in db])
                    all_scores.extend([float(x) for x in s])

                # pick topk
                arr = np.asarray(all_scores, dtype=np.float32)
                k = min(int(topk), int(arr.size))
                if k <= 0:
                    run[str(qid)] = {}
                    continue
                top_pos = np.argpartition(-arr, kth=k - 1)[:k]
                top_pos = top_pos[np.argsort(-arr[top_pos], kind="mergesort")]
                run[str(qid)] = {str(all_docids[p]): float(arr[p]) for p in top_pos.tolist()}

        self._last_token_stats = {
            "text_side": q_seq_stats.as_dict(),
            "image_side": d_seq_stats.as_dict(),
            "breakdown": {
                "text_query_total_seq": q_seq_stats.as_dict(),
                "image_doc_total_seq": d_seq_stats.as_dict(),
                "image_doc_rerank_seq": d_seq_stats_rerank.as_dict(),
            },
        }
        return run
