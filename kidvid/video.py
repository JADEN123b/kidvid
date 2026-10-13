import math
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1280, 720, 24
FADE = 0.4
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


@dataclass
class Caption:
    text: str
    start: float
    end: float


@dataclass
class Shot:
    image: Image.Image
    duration: float
    captions: list[Caption] = field(default_factory=list)
    clip: list[Image.Image] | None = None
    title: str | None = None


def caption_overlay(text: str) -> Image.Image:
    font = load_font(38)
    lines = textwrap.wrap(text, width=52)
    line_h = 48
    box_h = line_h * len(lines) + 30
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    widths = [draw.textlength(line, font=font) for line in lines]
    box_w = max(widths) + 60
    x0, y0 = (W - box_w) / 2, H - box_h - 28
    draw.rounded_rectangle((x0, y0, x0 + box_w, y0 + box_h), radius=22, fill=(0, 0, 0, 150))
    for i, (line, lw) in enumerate(zip(lines, widths)):
        draw.text(((W - lw) / 2, y0 + 15 + i * line_h), line, font=font, fill="white",
                  stroke_width=2, stroke_fill="black")
    return overlay


def title_card(image: Image.Image, title: str) -> Image.Image:
    bg = image.convert("RGB").resize((W, H), Image.LANCZOS).filter(ImageFilter.GaussianBlur(6))
    card = bg.convert("RGBA")
    draw = ImageDraw.Draw(card)
    font = load_font(76)
    lines = textwrap.wrap(title, width=24)
    total_h = 92 * len(lines)
    for i, line in enumerate(lines):
        lw = draw.textlength(line, font=font)
        draw.text(((W - lw) / 2, (H - total_h) / 2 + i * 92), line, font=font, fill=(255, 236, 140),
                  stroke_width=5, stroke_fill=(60, 30, 90))
    return card.convert("RGB")


def _ease(t: float) -> float:
    return t * t * (3 - 2 * t)


def ken_burns_frames(image: Image.Image, n: int, style: int):
    base = image.convert("RGB").resize((int(W * 1.25), int(H * 1.25)), Image.LANCZOS)
    bw, bh = base.size
    z0, z1 = [(1.0, 1.18), (1.18, 1.0), (1.12, 1.12), (1.12, 1.12)][style % 4]
    pan = [(0.5, 0.5, 0.5, 0.45), (0.5, 0.45, 0.5, 0.5), (0.3, 0.5, 0.7, 0.5), (0.7, 0.5, 0.3, 0.5)][style % 4]
    for i in range(n):
        t = _ease(i / max(n - 1, 1))
        z = z0 + (z1 - z0) * t
        cx = (pan[0] + (pan[2] - pan[0]) * t) * bw
        cy = (pan[1] + (pan[3] - pan[1]) * t) * bh
        cw, ch = bw / z, bh / z
        cx = min(max(cx, cw / 2), bw - cw / 2)
        cy = min(max(cy, ch / 2), bh - ch / 2)
        box = (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)
        yield base.transform((W, H), Image.EXTENT, box, Image.BILINEAR)


def clip_frames(clip: list[Image.Image], n: int, src_fps: int = 24):
    frames = [f.convert("RGB").resize((W, H), Image.LANCZOS) for f in clip]
    loop = frames + frames[-2:0:-1] if len(frames) > 2 else frames
    step = src_fps / FPS
    for i in range(n):
        yield loop[int(i * step) % len(loop)]


def render(shots: list[Shot], audio_wav: Path, out_path: Path, music: Path | None = None,
           music_volume: float = 0.12, progress=None) -> Path:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-i", str(audio_wav)]
    if music:
        cmd += ["-stream_loop", "-1", "-i", str(music), "-filter_complex",
                f"[2:a]volume={music_volume}[m];[1:a][m]amix=inputs=2:duration=first:normalize=0[a]",
                "-map", "0:v", "-map", "[a]"]
    else:
        cmd += ["-map", "0:v", "-map", "1:a"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    fade_n = int(FADE * FPS)
    try:
        for si, shot in enumerate(shots):
            n = frame_count(shot.duration)
            if shot.title:
                still = title_card(shot.image, shot.title)
                frames = (still for _ in range(n))
            elif shot.clip:
                frames = clip_frames(shot.clip, n)
            else:
                frames = ken_burns_frames(shot.image, n, si)
            overlays = [(c, caption_overlay(c.text)) for c in shot.captions]
            for i, frame in enumerate(frames):
                t = i / FPS
                active = next((o for c, o in overlays if c.start <= t < c.end), None)
                if active is not None:
                    frame = Image.alpha_composite(frame.convert("RGBA"), active).convert("RGB")
                arr = np.asarray(frame, dtype=np.uint8)
                fade = min(1.0, (i + 1) / fade_n, (n - i) / fade_n)
                if fade < 1.0:
                    arr = (arr.astype(np.float32) * fade).astype(np.uint8)
                proc.stdin.write(arr.tobytes())
            if progress:
                progress((si + 1) / len(shots))
    finally:
        proc.stdin.close()
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {code}")
    return out_path


def frame_count(duration: float) -> int:
    return max(1, math.ceil(duration * FPS))
