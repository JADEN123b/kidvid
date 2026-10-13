import ctypes
import gc

import torch


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def has_gpu() -> bool:
    return torch.cuda.is_available()


def dtype() -> torch.dtype:
    if has_gpu():
        return torch.float16
    try:
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            return torch.bfloat16
    except (AttributeError, RuntimeError):
        pass
    return torch.float32


def free_memory() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
