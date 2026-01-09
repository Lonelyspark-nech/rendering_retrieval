# -*- coding: utf-8 -*-
import os, json, time, hashlib
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Callable

import numpy as np

try:
    import faiss  # type: ignore
except Exception:
    faiss = None


def l2_normalize_inplace(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    x /= n
    return x


def sha1_docids(doc_ids: List[str]) -> str:
    h = hashlib.sha1()
    for d in doc_ids:
        h.update(d.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def sha1_pairs(pairs: Iterable[Tuple[str, str]]) -> str:
    """Stable signature for corpus contents (docid + payload string).

    Use this when doc_ids alone are not enough (e.g., same ids but different text/images).
    """
    h = hashlib.sha1()
    for a, b in pairs:
        h.update(str(a).encode("utf-8"))
        h.update(b"\t")
        h.update(("" if b is None else str(b)).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def cfg_key(cfg: Dict[str, Any]) -> str:
    s = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def paths(cache_dir: str, key: str) -> Tuple[str, str, str]:
    os.makedirs(cache_dir, exist_ok=True)
    return (
        os.path.join(cache_dir, f"flatip_{key}.faiss"),
        os.path.join(cache_dir, f"docids_{key}.json"),
        os.path.join(cache_dir, f"meta_{key}.json"),
    )


def load_index(
    cache_dir: str,
    cfg: Dict[str, Any],
    doc_ids: List[str],
    rebuild: bool = False,
    check_docids: bool = True,
    use_gpu: bool = False,
    gpu_device: int = 0,
):
    if faiss is None:
        raise ImportError("faiss not installed. Install faiss-cpu or faiss-gpu.")

    key = cfg_key(cfg)
    ipath, dpath, mpath = paths(cache_dir, key)
    if rebuild or not (os.path.exists(ipath) and os.path.exists(dpath) and os.path.exists(mpath)):
        return None

    meta = json.load(open(mpath, "r", encoding="utf-8"))
    if meta.get("cfg") != cfg:
        return None

    cached_docids = json.load(open(dpath, "r", encoding="utf-8"))
    if len(cached_docids) != len(doc_ids):
        return None
    if check_docids and meta.get("docids_sha1") != sha1_docids(doc_ids):
        return None

    index_cpu = faiss.read_index(ipath)
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, int(gpu_device), index_cpu)
    else:
        index = index_cpu
    return index, cached_docids, meta


def save_index(
    cache_dir: str,
    cfg: Dict[str, Any],
    index_cpu,
    doc_ids: List[str],
    extra_meta: Optional[Dict[str, Any]] = None,
):
    if faiss is None:
        raise ImportError("faiss not installed. Install faiss-cpu or faiss-gpu.")

    key = cfg_key(cfg)
    ipath, dpath, mpath = paths(cache_dir, key)

    faiss.write_index(index_cpu, ipath)
    json.dump(doc_ids, open(dpath, "w", encoding="utf-8"), ensure_ascii=False)
    meta = {
        "cfg": cfg,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ntotal": int(index_cpu.ntotal),
        "dim": int(index_cpu.d),
        "docids_sha1": sha1_docids(doc_ids),
    }
    if extra_meta:
        meta.update(extra_meta)
    json.dump(meta, open(mpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return ipath, dpath, mpath


def build_flatip(doc_vecs_f32: np.ndarray):
    """doc_vecs_f32: (N,D) float32, already L2-normalized if you want cosine."""
    if faiss is None:
        raise ImportError("faiss not installed. Install faiss-cpu or faiss-gpu.")
    dim = int(doc_vecs_f32.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(doc_vecs_f32)
    return index


def get_or_build_flatip(
    *,
    cache_dir: str,
    cfg: Dict[str, Any],
    doc_ids: List[str],
    encode_doc_batches: Callable[[], Iterator[np.ndarray]],
    rebuild: bool = False,
    check_docids: bool = True,
    use_gpu: bool = False,
    gpu_device: int = 0,
    normalize: bool = False,
    extra_meta: Optional[Any] = None,
):
    """Load cached FlatIP index or build it by streaming doc vectors.

    Args:
      encode_doc_batches: a *callable* that returns an iterator over (B,D) float32 numpy arrays.
        We make it a callable so it's only executed when a rebuild is needed.
      normalize: if True, L2-normalize vectors in-place before adding.

    Returns:
      (index, cached_doc_ids, meta, loaded)
    """
    loaded = False
    loaded_obj = load_index(
        cache_dir=cache_dir,
        cfg=cfg,
        doc_ids=doc_ids,
        rebuild=rebuild,
        check_docids=check_docids,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
    )
    if loaded_obj is not None:
        index, cached_docids, meta = loaded_obj
        return index, cached_docids, meta, True

    if faiss is None:
        raise ImportError("faiss not installed. Install faiss-cpu or faiss-gpu.")

    index_cpu = None
    n_added = 0
    for xb in encode_doc_batches():
        if xb is None:
            continue
        if not isinstance(xb, np.ndarray):
            xb = np.asarray(xb)
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32)
        if normalize:
            l2_normalize_inplace(xb)
        if index_cpu is None:
            dim = int(xb.shape[1])
            index_cpu = faiss.IndexFlatIP(dim)
        index_cpu.add(xb)
        n_added += int(xb.shape[0])

    if index_cpu is None or n_added == 0:
        raise ValueError("No doc vectors produced for index build.")

    # persist (allow lazy extra_meta computed after encoding)
    extra_meta_used = extra_meta() if callable(extra_meta) else extra_meta
    save_index(cache_dir=cache_dir, cfg=cfg, index_cpu=index_cpu, doc_ids=doc_ids, extra_meta=extra_meta_used)

    # optionally move to GPU for search
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, int(gpu_device), index_cpu)
    else:
        index = index_cpu

    meta = {"cfg": cfg, "ntotal": int(index_cpu.ntotal), "dim": int(index_cpu.d)}
    if extra_meta_used:
        meta.update(extra_meta_used)
    return index, doc_ids, meta, loaded
