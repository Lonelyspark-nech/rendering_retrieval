# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class QueryRecord:
    qid: str
    text: Optional[str] = None          # 原始 query 文本（LongEmbed queries.jsonl）
    image_path: Optional[str] = None    # 预留：未来你可能也会有 query image
    prompt: Optional[str] = None        # 预留：instruction / message
    meta: Dict[str, Any] = None

@dataclass
class DocRecord:
    docid: str
    text: Optional[str] = None          # rendered corpus.jsonl 里的截断文本
    image_path: Optional[str] = None    # rendered images/<docid>.png
    meta: Dict[str, Any] = None
