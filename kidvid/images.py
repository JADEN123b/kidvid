import torch
from diffusers import LCMScheduler, StableDiffusionPipeline
from PIL import Image

from .device import device, dtype, free_memory, has_gpu
from .reference import BASE_MODEL, NEGATIVE, attach_references, scene_prompt

LCM_LORA = "latent-consistency/lcm-lora-sdv1-5"


class Illustrator:
    def __init__(self, references: list[Image.Image] | None = None, reference_strength: float = 0.6):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL, dtype=dtype(), safety_checker=None, requires_safety_checker=False
        )
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.load_lora_weights(LCM_LORA, adapter_name="lcm")
        self.pipe.fuse_lora()
        self.pipe.to(device())
        self.pipe.set_progress_bar_config(disable=True)
        self.embeds = attach_references(self.pipe, references or [], reference_strength)
        self.width, self.height = (912, 512) if has_gpu() else (768, 432)
        self.steps = 6 if has_gpu() else 4

    def draw(self, picture: str, characters: str, setting: str, seed: int) -> Image.Image:
        kwargs = {"ip_adapter_image_embeds": self.embeds} if self.embeds is not None else {}
        return self.pipe(
            prompt=scene_prompt(picture, characters, setting, self.embeds is not None),
            negative_prompt=NEGATIVE,
            num_inference_steps=self.steps,
            guidance_scale=1.0,
            width=self.width,
            height=self.height,
            generator=torch.Generator("cpu").manual_seed(seed),
            **kwargs,
        ).images[0]

    def close(self) -> None:
        del self.pipe
        free_memory()
