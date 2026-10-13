"""Shared Stable Diffusion 1.5 setup: cartoon base model, fast LCM sampling, optional IP-Adapter references."""
from pathlib import Path

import torch
from PIL import Image

from .device import device

BASE_MODEL = "Lykon/dreamshaper-8"
IP_ADAPTER = ("h94/IP-Adapter", "models", "ip-adapter_sd15.safetensors")

STYLE = "cute cartoon children's picture book illustration, soft pastel colours, warm light"
NEGATIVE = "text, words, letters, watermark, scary, dark, violent, blurry, deformed, ugly"


def load_references(paths) -> list[Image.Image]:
    images = []
    for p in paths or []:
        p = getattr(p, "name", p)
        if isinstance(p, (tuple, list)):
            p = p[0]
        if p and Path(p).exists():
            images.append(Image.open(p).convert("RGB"))
    return images


def attach_references(pipe, references: list[Image.Image], strength: float):
    """Load IP-Adapter, encode the reference pictures once, then drop the image encoder to save memory."""
    if not references:
        return None
    pipe.load_ip_adapter(IP_ADAPTER[0], subfolder=IP_ADAPTER[1], weight_name=IP_ADAPTER[2])
    pipe.set_ip_adapter_scale(strength)
    pipe.image_encoder.to(device())
    with torch.no_grad():
        embeds = pipe.prepare_ip_adapter_image_embeds([references], None, device(), 1, False)
    pipe.image_encoder = None
    return embeds


def scene_prompt(picture: str, characters: str, setting: str, has_reference: bool) -> str:
    style = "" if has_reference else f"{STYLE}. "
    return f"{style}{picture}. {characters}. {setting}"
