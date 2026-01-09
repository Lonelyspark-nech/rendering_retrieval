# -*- coding: utf-8 -*-
from typing import Any, Dict

def _parse_dtype(dtype: Any):
    # adapter_kwargs 里可写 "bfloat16"/"float16"/"float32"
    import torch
    if dtype is None:
        return torch.bfloat16
    if isinstance(dtype, str):
        s = dtype.lower()
        if s in ["bf16", "bfloat16"]:
            return torch.bfloat16
        if s in ["fp16", "float16", "half"]:
            return torch.float16
        if s in ["fp32", "float32"]:
            return torch.float32
    return dtype  # 允许你直接传 torch.dtype（如果你用 python 调用而不是 CLI）

def build_adapter(name: str, kwargs: Dict[str, Any]):
    name = name.strip()
    kwargs = dict(kwargs or {})

    if name == "vlm2vec_mmeb_text2image":
        from .vlm2vec_mmeb_adapter import VLM2VecMMEBText2ImageAdapter
        if "dtype" in kwargs:
            kwargs["dtype"] = _parse_dtype(kwargs["dtype"])
        return VLM2VecMMEBText2ImageAdapter(**kwargs)

    if name == "llm2vec_text":
        # 让 adapter 自己负责从 kwargs 构建底层 l2v + tokenizer + model
        from .llm2vec_adapter import LLM2VecTextAdapter
        return LLM2VecTextAdapter.from_kwargs(kwargs)
    if name == "vlm2vec_mmeb_textonly":
        from .vlm2vec_mmeb_text_adapter import VLM2VecMMEBTextOnlyAdapter
        if "dtype" in kwargs:
            kwargs["dtype"] = _parse_dtype(kwargs["dtype"])
        return VLM2VecMMEBTextOnlyAdapter(**kwargs)
    if name == "qwen3_embedding_8b":
        from .qwen3_embedding_8b_adapter import Qwen3Embedding8BAdapter
        return Qwen3Embedding8BAdapter.from_kwargs(kwargs)
    if name == "colpali":
        from .colpali_adapter import ColPaliAdapter
        if "dtype" in kwargs:
            kwargs["dtype"] = _parse_dtype(kwargs["dtype"])
        return ColPaliAdapter(**kwargs)
    if name in {"e5_v", "e5-v", "e5v"}:
        from .e5v_adapter import E5VAdapter
        if "dtype" in kwargs:
            kwargs["dtype"] = _parse_dtype(kwargs["dtype"])
        return E5VAdapter(**kwargs)
    if name in {"vlm2vec_llava_next_text2image", "vlm2vec_llava_next"}:
        from .vlm2vec_llava_next_adapter import VLM2VecLlavaNextText2ImageAdapter
        if "dtype" in kwargs:
            kwargs["dtype"] = _parse_dtype(kwargs["dtype"])
        return VLM2VecLlavaNextText2ImageAdapter(**kwargs)

    raise ValueError(f"Unknown adapter: {name}. Available: vlm2vec_mmeb_text2image, llm2vec_text, vlm2vec_mmeb_textonly")
