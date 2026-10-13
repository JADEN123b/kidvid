import re

import numpy as np
from kokoro import KPipeline

SAMPLE_RATE = 24000

VOICES = {
    "Heart (warm female, US)": ("a", "af_heart"),
    "Bella (bright female, US)": ("a", "af_bella"),
    "Nicole (soft female, US)": ("a", "af_nicole"),
    "Michael (calm male, US)": ("a", "am_michael"),
    "Fenrir (deep male, US)": ("a", "am_fenrir"),
    "Emma (female, UK)": ("b", "bf_emma"),
    "George (male, UK)": ("b", "bm_george"),
}


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


class Narrator:
    def __init__(self, voice: str = "Heart (warm female, US)", speed: float = 0.9):
        lang, self.voice = VOICES.get(voice, ("a", voice))
        self.speed = speed
        self.pipe = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M")

    def speak(self, sentence: str) -> np.ndarray:
        chunks = [
            audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
            for _, _, audio in self.pipe(sentence, voice=self.voice, speed=self.speed)
            if audio is not None
        ]
        if not chunks:
            return np.zeros(int(0.3 * SAMPLE_RATE), dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32)
