# -*- coding: utf-8 -*-
import os
from typing import Dict, List, Tuple, Optional
from tqdm.auto import tqdm

from ..records import QueryRecord, DocRecord
from ..utils import iter_jsonl, find_image_for_doc

_LONGEMBED_DATASETS = ["2wikimqa", "narrativeqa", "qmsum", "summ_screen_fd"]

def _pick_first(d: dict, keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in d and d[k] is not None:
            return str(d[k])
    return None

def load_longembed_queries(queries_jsonl: str) -> List[QueryRecord]:
    out: List[QueryRecord] = []
    for ex in iter_jsonl(queries_jsonl, desc=f"load queries: {os.path.basename(os.path.dirname(queries_jsonl))}"):
        qid = _pick_first(ex, ["qid", "query_id", "id", "q_id"])
        if qid is None:
            raise ValueError(f"Cannot find qid key in query item: keys={list(ex.keys())}")

        qtext = _pick_first(ex, ["text", "query", "question", "message"])
        # LongEmbed 一般都有 query/text；这里允许空，但模型可能会报错
        out.append(QueryRecord(qid=qid, text=qtext, prompt=None, image_path=None, meta=ex))
    return out

def load_longembed_qrels(qrels_jsonl: str) -> Dict[str, Dict[str, int]]:
    """
    尽量兼容多种 qrels 组织：
    - 每行: {"qid":..., "doc_id":..., "rel":...}
    - 每行: {"qid":..., "doc_ids":[...]} / {"qid":..., "positive_doc_ids":[...]}
    - 或者每行是 {qid: {...}} 这种映射（较少见）
    """
    qrels: Dict[str, Dict[str, int]] = {}
    for ex in iter_jsonl(qrels_jsonl, desc=f"load qrels: {os.path.basename(os.path.dirname(qrels_jsonl))}"):
        if isinstance(ex, dict) and "qid" in ex and ("doc_id" in ex or "docid" in ex):
            qid = str(ex["qid"])
            docid = str(ex.get("doc_id", ex.get("docid")))
            rel = ex.get("rel", ex.get("relevance", ex.get("score", 1)))
            qrels.setdefault(qid, {})[docid] = int(rel)
            continue

        if isinstance(ex, dict) and "qid" in ex and any(k in ex for k in ["doc_ids", "positive_doc_ids", "positives"]):
            qid = str(ex["qid"])
            lst = ex.get("doc_ids", ex.get("positive_doc_ids", ex.get("positives")))
            if isinstance(lst, dict):
                # e.g. {"docA":1, "docB":2}
                for docid, rel in lst.items():
                    qrels.setdefault(qid, {})[str(docid)] = int(rel)
            else:
                for docid in lst:
                    qrels.setdefault(qid, {})[str(docid)] = 1
            continue

        # 可能是 { "<qid>": { "<docid>": rel, ... } }
        if isinstance(ex, dict) and len(ex) == 1:
            qid = str(next(iter(ex.keys())))
            inner = ex[qid]
            if isinstance(inner, dict):
                for docid, rel in inner.items():
                    qrels.setdefault(qid, {})[str(docid)] = int(rel)
                continue

        raise ValueError(f"Unrecognized qrels format item: {ex}")
    return qrels

def load_rendered_corpus(rendered_dataset_dir: str, require_images: bool = False) -> List[DocRecord]:
    """
    rendered_dataset_dir:
      /.../longembed_rendered_v2/<setting>/<dataset>/
        corpus.jsonl
        images/<docid>.png
    """
    corpus_jsonl = os.path.join(rendered_dataset_dir, "corpus.jsonl")
    images_dir = os.path.join(rendered_dataset_dir, "images")

    out: List[DocRecord] = []
    missing_images = 0

    for ex in iter_jsonl(corpus_jsonl, desc=f"load corpus: {os.path.basename(rendered_dataset_dir)}"):
        docid = str(ex.get("doc_id", ex.get("docid", ex.get("id"))))
        if docid is None:
            raise ValueError(f"Cannot find doc_id key in corpus item: keys={list(ex.keys())}")
        text = ex.get("text", None)
        img = find_image_for_doc(images_dir, docid) if os.path.isdir(images_dir) else None
        if require_images and img is None:
            missing_images += 1
            continue
        out.append(DocRecord(docid=docid, text=text, image_path=img, meta=ex))

    if require_images and missing_images > 0:
        print(f"[warn] require_images=True: skipped {missing_images} docs without images in {rendered_dataset_dir}")
    return out

def load_longembed_rendered_bundle(
    longembed_root: str,
    rendered_root: str,
    setting: str,
    dataset: str,
    require_doc_images: bool = False,
) -> Tuple[List[QueryRecord], List[DocRecord], Dict[str, Dict[str, int]]]:
    qpath = os.path.join(longembed_root, dataset, "queries.jsonl")
    rpath = os.path.join(longembed_root, dataset, "qrels.jsonl")
    rendered_dir = os.path.join(rendered_root, setting, dataset)

    queries = load_longembed_queries(qpath)
    qrels = load_longembed_qrels(rpath)
    corpus = load_rendered_corpus(rendered_dir, require_images=require_doc_images)
    return queries, corpus, qrels

def default_datasets() -> List[str]:
    return list(_LONGEMBED_DATASETS)
