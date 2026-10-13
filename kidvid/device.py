import gc

import torch


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def has_gpu() -> bool:
    return torch.cuda.is_available()


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
