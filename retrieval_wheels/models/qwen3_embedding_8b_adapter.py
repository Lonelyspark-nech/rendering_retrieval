# -*- coding: utf-8 -*-
from typing import Dict, List, Optional, Iterator
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import os

from .base import ModelAdapter
from ..records import QueryRecord, DocRecord
from ..utils import chunked, TokenStats
from ..index.faiss_flatip_cache import get_or_build_flatip, sha1_pairs, l2_normalize_inplace
from ..index.faiss_flatip_cache import get_or_build_flatip, sha1_pairs, l2_normalize_inplace


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    # same logic as HF example
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        seq_lens = attention_mask.sum(dim=1) - 1
        bsz = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(bsz, device=last_hidden_states.device), seq_lens]


class Qwen3Embedding8BAdapter(ModelAdapter):
    """
    Qwen3-Embedding-8B text-only adapter:
    - query: optional instruction prefix (we reuse the same "instruction" arg as LLM2Vec for fairness)
    - doc: plain text
    - pooling: last-token pooling
    - sim: cosine via normalize + matmul
    - token stats: attention_mask.sum()
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_length: int = 8192,
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
        model_name: Optional[str] = None,
        dtype: Optional[str] = None,
    ):
        self.model = model
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

        self.model_name = model_name or getattr(getattr(self.model, "config", None), "_name_or_path", None)
        self.dtype = dtype

    @property
    def name(self) -> str:
        return "qwen3_embedding_8b"

    def get_last_token_stats(self) -> dict:
        return getattr(self, "_last_token_stats", {})

    @classmethod
    def from_kwargs(cls, kwargs: dict) -> "Qwen3Embedding8BAdapter":
        model_name = kwargs.get("model_name", "Qwen/Qwen3-Embedding-8B")
        instruction = kwargs.get("instruction", None)
        max_length = int(kwargs.get("max_length", 8192))
        encode_bs = int(kwargs.get("encode_bs", 32))
        dtype = kwargs.get("dtype", "bf16")

        # faiss options
        use_faiss = bool(kwargs.get("use_faiss", True))
        faiss_cache_dir = kwargs.get("faiss_cache_dir", None)
        faiss_rebuild = bool(kwargs.get("faiss_rebuild", False))
        faiss_check_docids = bool(kwargs.get("faiss_check_docids", True))
        faiss_use_gpu = bool(kwargs.get("faiss_use_gpu", False))
        faiss_gpu_device = int(kwargs.get("faiss_gpu_device", 0))

        from transformers import AutoTokenizer, AutoModel
        import torch

        # dtype
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_map.get(str(dtype).lower(), torch.bfloat16)

        # tokenizer MUST be left padding for last-token pooling correctness
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            trust_remote_code=True,
            use_fast=False,          # 关键
        )

        # model
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )
        model.eval()
        return cls(
            model=model,
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
            model_name=model_name,
            dtype=str(dtype),
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
            # IMPORTANT: keep the same formatting as your LLM2Vec adapter: [instruction, query]
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
        """Stream doc embeddings as float32 numpy batches (updates doc token stats)."""
        for batch in tqdm(list(chunked(doc_texts, self.encode_bs)), desc="encode corpus (Qwen3Emb)", unit="batch"):
            batch_texts = list(batch)
            batch_dict = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            # token stats (docs)
            lens = batch_dict["attention_mask"].sum(dim=1).tolist()
            self._d_text_stats.add_many([int(x) for x in lens])

            batch_dict = {k: v.to(self.model.device) for k, v in batch_dict.items()}
            with torch.no_grad():
                out = self.model(**batch_dict)
                emb = last_token_pool(out.last_hidden_state, batch_dict["attention_mask"])
                emb = F.normalize(emb.float(), p=2, dim=1)
            xb = emb.detach().float().cpu().numpy().astype(np.float32, copy=False)
            yield xb

    def _encode_texts_batched(self, texts: List[str], desc: str) -> torch.Tensor:
        reps = []
        for batch in tqdm(list(chunked(texts, self.encode_bs)), desc=desc, unit="batch"):
            batch_dict = self.tokenizer(
                list(batch),
                padding=True,                # padding here is fine; tokenizer padding_side is left
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch_dict = {k: v.to(self.model.device) for k, v in batch_dict.items()}

            with torch.no_grad():
                out = self.model(**batch_dict)
                emb = last_token_pool(out.last_hidden_state, batch_dict["attention_mask"])
                emb = F.normalize(emb.float(), p=2, dim=1)

            reps.append(emb.cpu())
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
        for batch in tqdm(list(chunked(qtexts, self.encode_bs)), desc="tokenize queries (Qwen3Emb)", unit="batch"):
            self._q_text_stats.add_many(self._token_lens_queries(list(batch)))

        if not self.use_faiss:
            # legacy brute-force
            for batch in tqdm(list(chunked(doc_texts, self.encode_bs)), desc="tokenize corpus (Qwen3Emb)", unit="batch"):
                self._d_text_stats.add_many(self._token_lens_docs(list(batch)))

            d_reps = self._encode_texts_batched(doc_texts, desc="encode corpus (Qwen3Emb)")  # (nd, D)

            if self.instruction:
                q_inputs = [f"{self.instruction} {qt}" for qt in qtexts]
            else:
                q_inputs = qtexts
            q_reps = self._encode_texts_batched(q_inputs, desc="encode queries (Qwen3Emb)")  # (nq, D)

            q_reps_norm = F.normalize(q_reps.float(), p=2, dim=1)
            d_reps_norm = F.normalize(d_reps.float(), p=2, dim=1)

            run: Dict[str, Dict[str, float]] = {}
            for i in tqdm(range(q_reps_norm.size(0)), desc="scoring+topk (Qwen3Emb)", unit="q"):
                scores = torch.mv(d_reps_norm, q_reps_norm[i])  # (nd,)
                k = min(topk, scores.numel())
                vals, idx = torch.topk(scores, k=k, largest=True)
                run[qids[i]] = {doc_ids[j]: float(vals[t].item()) for t, j in enumerate(idx.tolist())}
        else:
            corpus_sig = self._corpus_sig(corpus)
            cfg = {
                "adapter": self.name,
                "model_name": self.model_name,
                "dtype": self.dtype,
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

            if self.instruction:
                q_inputs = [f"Instruct: {self.instruction}\nQuery:{qt}" for qt in qtexts]
            else:
                q_inputs = qtexts
            q_reps = self._encode_texts_batched(q_inputs, desc="encode queries (Qwen3Emb)")
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

        # aggregate token stats (same schema as LLM2VecTextAdapter)
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
