# -*- coding: utf-8 -*-
"""
Base interfaces for retrieval evaluation.

B-scheme: each model adapter implements `search()` and returns a TREC-style run dict:
    run[qid][docid] = score (float)

The evaluation runner stays model-agnostic:
- it loads queries/corpus/qrels via dataset adapters
- calls adapter.search(...)
- runs pytrec_eval
- writes metrics + token_stats

Model-specific details (tokenization, prompts, pooling, similarity, etc.) MUST stay inside adapters.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from ..records import QueryRecord, DocRecord


RunDict = Dict[str, Dict[str, float]]  # run[qid][docid] = score


class ModelAdapter(ABC):
    """
    Model adapter interface.

    Requirements:
    - `search()` must be deterministic given the same inputs (unless the underlying model is stochastic).
    - `search()` should NOT mutate input records.
    - Adapters should store per-run statistics (e.g., token lengths) and expose via `get_last_token_stats()`.
    """

    # ---- identity ----
    @property
    @abstractmethod
    def name(self) -> str:
        """A short stable identifier used for output folder/file naming."""
        raise NotImplementedError

    # ---- capability flags ----
    @property
    def requires_doc_images(self) -> bool:
        """
        Whether the adapter REQUIRES corpus docs to have image_path (e.g. image retrieval / image-as-doc).
        The dataset loader can use this to filter out docs without images.
        """
        return False

    @property
    def requires_query_images(self) -> bool:
        """
        Whether the adapter REQUIRES queries to have image_path (rare; e.g. image-to-text retrieval).
        If True, the dataset loader/runner may need to validate or filter queries.
        """
        return False

    # ---- run-level statistics ----
    def reset_stats(self) -> None:
        """
        Called at the start of each `search()` to reset internal counters/stats.
        Default: do nothing.
        """
        return None

    def get_last_token_stats(self) -> Dict[str, Any]:
        """
        Return per-run token length stats to be saved alongside trec_eval metrics.

        Recommended schema (you can extend it):
        {
          "text_side":  {"count":..., "min":..., "max":..., "avg":...},
          "image_side": {"count":..., "min":..., "max":..., "avg":...},
          "breakdown": {... optional ...}
        }

        Default: empty dict (no stats).
        """
        return {}

    def get_last_extra_stats(self) -> Dict[str, Any]:
        """
        Optional hook for other per-run statistics (timings, memory, warnings, etc).
        Default: empty dict.
        """
        return {}

    # ---- main API ----
    @abstractmethod
    def search(
        self,
        queries: List[QueryRecord],
        corpus: List[DocRecord],
        topk: int,
    ) -> RunDict:
        """
        Perform retrieval and return a TREC-style run dictionary.

        Args:
            queries: list of QueryRecord (qid must be non-empty)
            corpus: list of DocRecord (docid must be non-empty)
            topk: retrieve at most topk docs per query

        Returns:
            run: Dict[qid, Dict[docid, score]]

        Notes:
            - Scores can be any real number; higher means more relevant.
            - If a query yields fewer than topk docs (e.g., corpus smaller), return as many as available.
            - Keys should be strings; ensure qid/docid are str in output.
        """
        raise NotImplementedError

    # ---- optional validation helpers ----
    def validate_inputs(self, queries: List[QueryRecord], corpus: List[DocRecord]) -> None:
        """
        Optional input validation. Adapters may override for stricter checks.
        Default: minimal checks.
        """
        for q in queries:
            if not q.qid:
                raise ValueError("Encountered query with empty qid.")
            if self.requires_query_images and not q.image_path:
                raise ValueError(f"Query requires image_path but missing for qid={q.qid}")
        for d in corpus:
            if not d.docid:
                raise ValueError("Encountered doc with empty docid.")
            if self.requires_doc_images and not d.image_path:
                raise ValueError(f"Doc requires image_path but missing for docid={d.docid}")
