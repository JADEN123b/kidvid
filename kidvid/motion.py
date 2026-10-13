import torch
from PIL import Image

from .device import free_memory, has_gpu

MODEL = "Lightricks/LTX-Video"
WIDTH, HEIGHT, NUM_FRAMES, FPS = 704, 480, 97, 24
NEGATIVE = "worst quality, inconsistent motion, blurry, jittery, distorted, text, watermark"


class Animator:
    """Image-to-video with LTX-Video. GPU only (works on a free Colab T4)."""

    def __init__(self):
        if not has_gpu():
            raise RuntimeError("AI motion mode needs an NVIDIA GPU (e.g. free Google Colab T4).")
        from diffusers import LTXImageToVideoPipeline

        self.pipe = LTXImageToVideoPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_tiling()
        self.pipe.set_progress_bar_config(disable=True)

    def animate(self, image: Image.Image, picture: str, seed: int, steps: int = 30) -> list[Image.Image]:
        prompt = f"{picture}. Gentle cartoon animation, smooth natural movement, children's picture book style."
        generator = torch.Generator("cpu").manual_seed(seed)
        return self.pipe(
            image=image.resize((WIDTH, HEIGHT), Image.LANCZOS),
            prompt=prompt,
            negative_prompt=NEGATIVE,
            width=WIDTH,
            height=HEIGHT,
            num_frames=NUM_FRAMES,
            num_inference_steps=steps,
            generator=generator,
        ).frames[0]

    def close(self) -> None:
        del self.pipe
        free_memory()
