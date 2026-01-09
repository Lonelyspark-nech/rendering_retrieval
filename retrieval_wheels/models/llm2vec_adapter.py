# -*- coding: utf-8 -*-
from typing import Dict, List, Optional, Iterator
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import os

from .base import ModelAdapter
from ..records import QueryRecord, DocRecord
from ..utils import chunked, TokenStats
from ..index.faiss_flatip_cache import get_or_build_flatip, sha1_pairs, l2_normalize_inplace

class LLM2VecTextAdapter(ModelAdapter):
    """
    - encode: 复用 l2v.encode
    - sim: 复用 normalize + matmul (cosine)
    - token stats: 复用 tokenizer 统计 attention_mask.sum()
    """
    def __init__(
        self,
        l2v,
        tokenizer,
        max_length: int = 512,
        instruction: Optional[str] = None,
        encode_bs: int = 32,
        # ---- faiss (FlatIP) index ----
        use_faiss: bool = True,
        faiss_cache_dir: Optional[str] = None,
        faiss_rebuild: bool = False,
        faiss_check_docids: bool = True,
        faiss_use_gpu: bool = False,
        faiss_gpu_device: int = 0,
        # adapter identity for cache keys
        model_dir: Optional[str] = None,
        pooling_mode: str = "mean",
    ):
        self.l2v = l2v
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.instruction = instruction
        self.encode_bs = int(encode_bs)

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

        self.model_dir = model_dir
        self.pooling_mode = str(pooling_mode)

    @property
    def name(self) -> str:
        return "llm2vec_text"

    def get_last_token_stats(self) -> dict:
        return getattr(self, "_last_token_stats", {})

    @classmethod
    def from_kwargs(cls, kwargs: dict) -> "LLM2VecTextAdapter":
        model_dir = kwargs.get("model_dir")
        if not model_dir:
            raise ValueError("llm2vec_text requires adapter_kwargs.model_dir")

        instruction = kwargs.get("instruction", None)
        pooling_mode = kwargs.get("pooling_mode", "mean")
        max_length = int(kwargs.get("max_length", 512))
        encode_bs = int(kwargs.get("encode_bs", 32))
        # faiss options
        use_faiss = bool(kwargs.get("use_faiss", True))
        faiss_cache_dir = kwargs.get("faiss_cache_dir", None)
        faiss_rebuild = bool(kwargs.get("faiss_rebuild", False))
        faiss_check_docids = bool(kwargs.get("faiss_check_docids", True))
        faiss_use_gpu = bool(kwargs.get("faiss_use_gpu", False))
        faiss_gpu_device = int(kwargs.get("faiss_gpu_device", 0))

        from llm2vec import LLM2Vec
        import torch
        from transformers import AutoTokenizer, AutoModel, AutoConfig
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_dir,
            trust_remote_code=True,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )
        model = PeftModel.from_pretrained(model, model_dir)

        l2v = LLM2Vec(model, tokenizer, pooling_mode=pooling_mode, max_length=max_length)
        return cls(
            l2v=l2v,
            tokenizer=tokenizer,
            max_length=max_length,
            instruction=instruction,
            encode_bs=encode_bs,
            use_faiss=use_faiss,
            faiss_cache_dir=faiss_cache_dir,
            faiss_rebuild=faiss_rebuild,
            faiss_check_docids=faiss_check_docids,
            faiss_use_gpu=faiss_use_gpu,
            faiss_gpu_device=faiss_gpu_device,
            model_dir=model_dir,
            pooling_mode=pooling_mode,
        )

    def _token_lens_docs(self, texts: List[str]) -> List[int]:
        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            add_special_tokens=True,
        )
        return [len(ids) for ids in enc["input_ids"]]

    def _token_lens_queries(self, qtexts: List[str]) -> List[int]:
        if self.instruction:
            enc = self.tokenizer(
                [self.instruction] * len(qtexts),
                qtexts,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                add_special_tokens=True,
            )
        else:
            enc = self.tokenizer(
                qtexts,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                add_special_tokens=True,
            )
        return [len(ids) for ids in enc["input_ids"]]

    def _corpus_sig(self, corpus: List[DocRecord]) -> str:
        return sha1_pairs((d.docid, d.text or "") for d in corpus)

    def _iter_doc_vecs_f32(self, doc_texts: List[str]) -> Iterator[np.ndarray]:
        """Stream doc embeddings as float32 numpy batches."""
        for batch in tqdm(list(chunked(doc_texts, self.encode_bs)), desc="encode corpus (LLM2Vec)", unit="batch"):
            # token stats (docs)
            self._d_text_stats.add_many(self._token_lens_docs(list(batch)))
            r = self.l2v.encode(list(batch))
            if isinstance(r, torch.Tensor):
                xb = r.detach().float().cpu().numpy()
            else:
                xb = np.asarray(r, dtype=np.float32)
            xb = xb.astype(np.float32, copy=False)
            yield xb


    def _encode_batched(self, items: List, desc: str) -> torch.Tensor:
        reps = []
        for batch in tqdm(list(chunked(items, self.encode_bs)), desc=desc, unit="batch"):
            r = self.l2v.encode(batch)
            if isinstance(r, torch.Tensor):
                reps.append(r)
            else:
                reps.append(torch.tensor(r))
        return torch.cat(reps, dim=0)

    def search(self, queries: List[QueryRecord], corpus: List[DocRecord], topk: int) -> Dict[str, Dict[str, float]]:
        # reset token stats
        self._q_text_stats = TokenStats()
        self._d_text_stats = TokenStats()

        doc_ids = [d.docid for d in corpus]
        doc_texts = [d.text or "" for d in corpus]
        qids = [q.qid for q in queries]
        qtexts = [q.text or "" for q in queries]

        # token stats (queries)
        for batch in tqdm(list(chunked(qtexts, self.encode_bs)), desc="tokenize queries (LLM2Vec)", unit="batch"):
            self._q_text_stats.add_many(self._token_lens_queries(list(batch)))

        if not self.use_faiss:
            # legacy brute-force
            # token stats (docs)
            for batch in tqdm(list(chunked(doc_texts, self.encode_bs)), desc="tokenize corpus (LLM2Vec)", unit="batch"):
                self._d_text_stats.add_many(self._token_lens_docs(list(batch)))

            d_reps = self._encode_batched(doc_texts, desc="encode corpus (LLM2Vec)")
            if self.instruction:
                q_inputs = [[self.instruction, qt] for qt in qtexts]
            else:
                q_inputs = qtexts
            q_reps = self._encode_batched(q_inputs, desc="encode queries (LLM2Vec)")
            q_reps_norm = F.normalize(q_reps.float(), p=2, dim=1)
            d_reps_norm = F.normalize(d_reps.float(), p=2, dim=1)
            run: Dict[str, Dict[str, float]] = {}
            for i in tqdm(range(q_reps_norm.size(0)), desc="scoring+topk (LLM2Vec)", unit="q"):
                scores = torch.mv(d_reps_norm, q_reps_norm[i])
                k = min(topk, scores.numel())
                vals, idx = torch.topk(scores, k=k, largest=True)
                run[qids[i]] = {doc_ids[j]: float(vals[t].item()) for t, j in enumerate(idx.tolist())}
        else:
            corpus_sig = self._corpus_sig(corpus)
            cfg = {
                "adapter": self.name,
                "model_dir": self.model_dir,
                "pooling_mode": self.pooling_mode,
                "max_length": int(self.max_length),
                "instruction": self.instruction,
                "corpus_sig": corpus_sig,
                "index": "IndexFlatIP",
                "sim": "cosine",
            }

            extra_meta = lambda: {
                "doc_text_stats_raw": self._d_text_stats.to_raw(),
            }

            index, cached_docids, meta, loaded = get_or_build_flatip(
                cache_dir=self.faiss_cache_dir,
                cfg=cfg,
                doc_ids=doc_ids,
                encode_doc_batches=lambda: self._iter_doc_vecs_f32(doc_texts),
                rebuild=self.faiss_rebuild,
                check_docids=self.faiss_check_docids,
                use_gpu=self.faiss_use_gpu,
                gpu_device=self.faiss_gpu_device,
                normalize=True,  # cosine
                extra_meta=extra_meta,
            )

            if loaded and meta:
                self._d_text_stats = TokenStats.from_raw(meta.get("doc_text_stats_raw", {}))

            # encode queries
            if self.instruction:
                q_inputs = [[self.instruction, qt] for qt in qtexts]
            else:
                q_inputs = qtexts

            q_reps = self._encode_batched(q_inputs, desc="encode queries (LLM2Vec)")
            qv = q_reps.detach().float().cpu().numpy().astype(np.float32, copy=False)
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

        # aggregate token stats
        text_total = TokenStats()
        text_total.merge(self._q_text_stats)
        text_total.merge(self._d_text_stats)

        self._last_token_stats = {
            "text_side": text_total.as_dict(),
            "image_side": {"count": 0, "min": 0, "max": 0, "avg": 0.0},
            "breakdown": {
                "text_query": self._q_text_stats.as_dict(),
                "text_doc": self._d_text_stats.as_dict(),
            }
        }
        return run
