# -*- coding: utf-8 -*-
from typing import Dict, List, Optional, Iterator
from tqdm.auto import tqdm
import torch
import numpy as np
import os

from .base import ModelAdapter
from ..records import QueryRecord, DocRecord
from ..utils import chunked, TokenStats
from ..index.faiss_flatip_cache import get_or_build_flatip, sha1_pairs, l2_normalize_inplace

from src.model import MMEBModel
from src.arguments import ModelArguments
from src.model_utils import load_processor


class VLM2VecMMEBTextOnlyAdapter(ModelAdapter):
    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        qry_bs: int = 32,
        doc_bs: int = 32,
        use_tgt_for_query: bool = False,
        max_length: int = 4096,
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
        self.qry_bs = qry_bs
        self.doc_bs = doc_bs
        self.use_tgt_for_query = use_tgt_for_query
        self.max_length = int(max_length)

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
            model_name="/data3/sunbo/models/Qwen/Qwen2-VL-7B-Instruct",
            checkpoint_path="/data3/sunbo/models/TIGER-Lab/VLM2Vec-Qwen2VL-7B",
            pooling="last",
            normalize=True,
            model_backbone="qwen2_vl",
            lora=True,
        )
        self.processor = load_processor(self.model_args)
        self.model = MMEBModel.load(self.model_args).to(self.device, dtype=self.dtype).eval()

        self.model_name = self.model_args.model_name
        self.checkpoint_path = self.model_args.checkpoint_path

    @property
    def name(self) -> str:
        suffix = "tgtq" if self.use_tgt_for_query else "qryq"
        return f"vlm2vec_mmeb_textonly_{suffix}"

    def get_last_token_stats(self) -> dict:
        return getattr(self, "_last_token_stats", {})

    @torch.no_grad()
    def _encode_queries(self, queries: List[QueryRecord]) -> torch.Tensor:
        reps = []
        texts = [q.text or "" for q in queries]
        for batch in tqdm(list(chunked(texts, self.qry_bs)), desc="encode queries text (MMEB text-only)", unit="batch"):
            inputs = self.processor(
                text=list(batch),
                images=None,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            if "attention_mask" in inputs:
                lens = inputs["attention_mask"].sum(dim=1).detach().cpu().to(torch.int32).tolist()
                self._query_total_stats.add_many([int(x) for x in lens])

            inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=(self.device.startswith("cuda"))):
                if self.use_tgt_for_query:
                    out = self.model(tgt=inputs)["tgt_reps"]
                else:
                    out = self.model(qry=inputs)["qry_reps"]
            reps.append(out)
        return torch.cat(reps, dim=0)

    @torch.no_grad()
    def _encode_docs(self, corpus: List[DocRecord]) -> torch.Tensor:
        reps = []
        texts = [d.text or "" for d in corpus]
        for batch in tqdm(list(chunked(texts, self.doc_bs)), desc="encode corpus text (MMEB text-only)", unit="batch"):
            inputs = self.processor(
                text=list(batch),
                images=None,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            if "attention_mask" in inputs:
                lens = inputs["attention_mask"].sum(dim=1).detach().cpu().to(torch.int32).tolist()
                self._doc_total_stats.add_many([int(x) for x in lens])

            inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=(self.device.startswith("cuda"))):
                out = self.model(tgt=inputs)["tgt_reps"]
            reps.append(out)
        return torch.cat(reps, dim=0)

    def _corpus_sig(self, corpus: List[DocRecord]) -> str:
        return sha1_pairs((d.docid, d.text or "") for d in corpus)

    @torch.no_grad()
    def _iter_doc_vecs_f32(self, corpus: List[DocRecord]) -> Iterator[np.ndarray]:
        texts = [d.text or "" for d in corpus]
        for batch in tqdm(list(chunked(texts, self.doc_bs)), desc="encode corpus text (MMEB text-only)", unit="batch"):
            inputs = self.processor(
                text=list(batch),
                images=None,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            if "attention_mask" in inputs:
                lens = inputs["attention_mask"].sum(dim=1).detach().cpu().to(torch.int32).tolist()
                self._doc_total_stats.add_many([int(x) for x in lens])
            inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=(self.device.startswith("cuda"))):
                out = self.model(tgt=inputs)["tgt_reps"]
            xb = out.detach().float().cpu().numpy().astype(np.float32, copy=False)
            if self.faiss_normalize:
                l2_normalize_inplace(xb)
            yield xb

    def search(self, queries: List[QueryRecord], corpus: List[DocRecord], topk: int) -> Dict[str, Dict[str, float]]:
        self._query_total_stats = TokenStats()
        self._doc_total_stats = TokenStats()

        qids = [q.qid for q in queries]
        doc_ids = [d.docid for d in corpus]

        if not self.use_faiss:
            # legacy brute-force (small corpora only)
            q_reps = self._encode_queries(queries)   # (nq, D)
            d_reps = self._encode_docs(corpus)       # (nd, D)
            if self.use_tgt_for_query:
                sim = (q_reps @ d_reps.transpose(0, 1))
            else:
                sim = self.model.compute_similarity(q_reps, d_reps)
            scores = sim.float().cpu()
            run: Dict[str, Dict[str, float]] = {}
            for i in tqdm(range(scores.size(0)), desc="scoring+topk (MMEB text-only)", unit="q"):
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
                "max_length": int(self.max_length),
                "use_tgt_for_query": bool(self.use_tgt_for_query),
                "dtype": str(self.dtype),
                "corpus_sig": corpus_sig,
                "index": "IndexFlatIP",
            }

            extra_meta = lambda: {
                "doc_total_stats_raw": self._doc_total_stats.to_raw(),
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
                normalize=False,
                extra_meta=extra_meta,
            )

            if loaded and meta:
                self._doc_total_stats = TokenStats.from_raw(meta.get("doc_total_stats_raw", {}))

            q_reps = self._encode_queries(queries)
            qv = q_reps.detach().float().cpu().numpy().astype(np.float32, copy=False)
            if self.faiss_normalize:
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

        # B 口径输出（text-only 没有 image）
        merged_text = TokenStats()
        merged_text.merge(self._query_total_stats)
        merged_text.merge(self._doc_total_stats)

        self._last_token_stats = {
            "text_side": merged_text.as_dict(),
            "image_side": {"count": 0, "min": 0, "max": 0, "avg": 0.0},
            "breakdown": {
                "query_total_tokens": self._query_total_stats.as_dict(),
                "doc_total_tokens": self._doc_total_stats.as_dict(),
            },
        }
        return run
