import torch
from diffusers import (
    AnimateDiffVideoToVideoPipeline,
    LCMScheduler,
    MotionAdapter,
    UNet2DConditionModel,
    UNetMotionModel,
)
from PIL import Image

from .device import device, dtype, free_memory, has_gpu
from .reference import BASE_MODEL, NEGATIVE, attach_references, scene_prompt

MOTION_MODEL = "wangfuyun/AnimateLCM"
MOTION_LORA = "AnimateLCM_sd15_t2v_lora.safetensors"
FPS = 8


def _drift(image: Image.Image, n: int, w: int, h: int, zoom: float = 0.06) -> list[Image.Image]:
    """A tiny slow push-in on the still, used as the starting video so the animation keeps the picture's layout."""
    base = image.convert("RGB").resize((w, h), Image.LANCZOS)
    frames = []
    for i in range(n):
        z = 1.0 + zoom * i / max(n - 1, 1)
        cw, ch = w / z, h / z
        box = ((w - cw) / 2, (h - ch) / 2, (w + cw) / 2, (h + ch) / 2)
        frames.append(base.transform((w, h), Image.EXTENT, box, Image.BICUBIC))
    return frames


def _motion_unet(adapter: MotionAdapter) -> UNetMotionModel:
    """Build the motion UNet directly in half precision (diffusers would build it in float32 first: ~5 GB extra RAM)."""
    unet2d = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet", dtype=dtype())
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype())
    try:
        unet = UNetMotionModel.from_unet2d(unet2d, adapter)
    finally:
        torch.set_default_dtype(previous)
    del unet2d
    free_memory()
    return unet


class Animator:
    """Turns each scene picture into a short animated clip (AnimateDiff + AnimateLCM), keeping its look."""

    def __init__(self, references: list[Image.Image] | None = None, reference_strength: float = 0.6):
        adapter = MotionAdapter.from_pretrained(MOTION_MODEL, dtype=dtype())
        unet = _motion_unet(adapter)
        self.pipe = AnimateDiffVideoToVideoPipeline.from_pretrained(
            BASE_MODEL, unet=unet, motion_adapter=adapter, dtype=dtype()
        )
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config, beta_schedule="linear")
        self.pipe.load_lora_weights(MOTION_MODEL, weight_name=MOTION_LORA, adapter_name="lcm-lora")
        self.pipe.set_adapters(["lcm-lora"], [0.8])
        self.pipe.vae.enable_slicing()
        self.pipe.to(device())
        self.pipe.set_progress_bar_config(disable=True)
        self.embeds = attach_references(self.pipe, references or [], reference_strength)
        self.width, self.height = (768, 432) if has_gpu() else (512, 288)
        self.frames = 16

    def animate(self, image: Image.Image, picture: str, characters: str, setting: str, seed: int,
                strength: float = 0.6, steps: int = 8) -> list[Image.Image]:
        kwargs = {"ip_adapter_image_embeds": self.embeds} if self.embeds is not None else {}
        prompt = scene_prompt(picture, characters, setting, self.embeds is not None) + ", gentle lively motion"
        return self.pipe(
            video=_drift(image, self.frames, self.width, self.height),
            prompt=prompt,
            negative_prompt=NEGATIVE,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=1.0,
            generator=torch.Generator("cpu").manual_seed(seed),
            decode_chunk_size=4,
            **kwargs,
        ).frames[0]

    def close(self) -> None:
        del self.pipe
        free_memory()
