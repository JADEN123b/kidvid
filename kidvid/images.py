import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image

from .device import device, free_memory, has_gpu

CPU_MODEL = "stabilityai/sd-turbo"
GPU_MODEL = "stabilityai/sdxl-turbo"

STYLE = "cute cartoon children's picture book illustration, soft pastel colours, warm light"
NEGATIVE = "text, words, letters, watermark, scary, dark, violent, blurry, deformed, ugly"


class Illustrator:
    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or (GPU_MODEL if has_gpu() else CPU_MODEL)
        dtype = torch.float16 if has_gpu() else torch.float32
        kwargs = {"variant": "fp16"} if has_gpu() else {}
        self.pipe = AutoPipelineForText2Image.from_pretrained(self.model_id, dtype=dtype, **kwargs)
        self.pipe = self.pipe.to(device())
        self.pipe.set_progress_bar_config(disable=True)
        if has_gpu():
            self.width, self.height, self.steps = 1024, 576, 4
        else:
            self.width, self.height, self.steps = 768, 432, 2

    def draw(self, picture: str, characters: str, setting: str, seed: int) -> Image.Image:
        prompt = f"{STYLE}. {picture}. {characters}. {setting}"
        generator = torch.Generator("cpu").manual_seed(seed)
        return self.pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            num_inference_steps=self.steps,
            guidance_scale=0.0,
            width=self.width,
            height=self.height,
            generator=generator,
        ).images[0]

    def close(self) -> None:
        del self.pipe
        free_memory()
