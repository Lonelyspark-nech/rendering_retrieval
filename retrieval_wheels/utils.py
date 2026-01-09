# -*- coding: utf-8 -*-
import json
import os
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from tqdm.auto import tqdm

def iter_jsonl(path: str, desc: str) -> Iterator[Any]:
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=desc, unit="line"):
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def load_json_or_jsonl(path: str, desc: str) -> Any:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # default: jsonl
    return list(iter_jsonl(path, desc=desc))

def chunked(seq: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    for i in range(0, len(seq), batch_size):
        yield seq[i:i+batch_size]

def find_image_for_doc(images_dir: str, docid: str) -> Optional[str]:
    # 优先 png，其次 jpg/jpeg/webp
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = os.path.join(images_dir, f"{docid}{ext}")
        if os.path.exists(p):
            return p
    return None

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

class TokenStats:
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.min = None
        self.max = None

    def add_many(self, lens):
        for x in lens:
            x = int(x)
            self.count += 1
            self.sum += x
            self.min = x if self.min is None else min(self.min, x)
            self.max = x if self.max is None else max(self.max, x)

    def merge(self, other: "TokenStats"):
        if other is None or other.count == 0:
            return
        self.count += other.count
        self.sum += other.sum
        self.min = other.min if self.min is None else (other.min if other.min is not None and other.min < self.min else self.min)
        self.max = other.max if self.max is None else (other.max if other.max is not None and other.max > self.max else self.max)

    def to_raw(self) -> Dict[str, Any]:
        """Lossless serialization for caching (keeps `sum` so we can merge later)."""
        return {
            "count": int(self.count),
            "sum": int(self.sum),
            "min": None if self.min is None else int(self.min),
            "max": None if self.max is None else int(self.max),
        }

    @classmethod
    def from_raw(cls, raw: Dict[str, Any]) -> "TokenStats":
        ts = cls()
        if not raw:
            return ts
        ts.count = int(raw.get("count", 0))
        ts.sum = int(raw.get("sum", 0))
        mn = raw.get("min", None)
        mx = raw.get("max", None)
        ts.min = None if mn is None else int(mn)
        ts.max = None if mx is None else int(mx)
        return ts

    def as_dict(self):
        return {
            "count": int(self.count),
            "min": 0 if self.min is None else int(self.min),
            "max": 0 if self.max is None else int(self.max),
            "avg": 0.0 if self.count == 0 else float(self.sum / self.count),
        }