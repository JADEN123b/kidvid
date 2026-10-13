import re
import time
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from .device import has_gpu
from .reference import load_references
from .story import Scene, Story, release_writer, write_story
from .tts import SAMPLE_RATE, Narrator, split_sentences
from .video import FPS, Caption, Shot, frame_count, render

PAUSE = 0.35
LEAD_IN = 0.5
TAIL = 0.8
MIN_SECONDS = 90.0
MAX_EXTRA_PAUSE = 4.0

Progress = Callable[[float, str], None]


def story_from_text(text: str, title: str = "My Story", characters: str = "", setting: str = "") -> Story:
    """Turn plain story text into scenes: one scene per paragraph (or every ~3 sentences)."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if len(paragraphs) <= 1:
        sentences = split_sentences(text)
        paragraphs = [" ".join(sentences[i : i + 3]) for i in range(0, len(sentences), 3)]
    scenes = [Scene(narration=p, picture=p[:200]) for p in paragraphs]
    return Story(title=title, characters=characters, setting=setting, scenes=scenes)


def _narrate(narrator: Narrator, text: str) -> tuple[np.ndarray, list[Caption]]:
    pieces = [np.zeros(int(LEAD_IN * SAMPLE_RATE), dtype=np.float32)]
    captions, t = [], LEAD_IN
    for sentence in split_sentences(text):
        audio = narrator.speak(sentence)
        dur = len(audio) / SAMPLE_RATE
        captions.append(Caption(sentence, t, t + dur + PAUSE))
        pieces += [audio, np.zeros(int(PAUSE * SAMPLE_RATE), dtype=np.float32)]
        t += dur + PAUSE
    pieces.append(np.zeros(int(TAIL * SAMPLE_RATE), dtype=np.float32))
    return np.concatenate(pieces), captions


def _pad_to_frames(audio: np.ndarray) -> tuple[np.ndarray, float]:
    duration = len(audio) / SAMPLE_RATE
    n = frame_count(duration)
    target = round(n / FPS * SAMPLE_RATE)
    return np.pad(audio, (0, max(0, target - len(audio))))[:target], n / FPS


def _stretch_to_minimum(tracks: list, n: int) -> list:
    """Give each scene a little extra breathing room so a full video lasts at least MIN_SECONDS."""
    total = sum(t[1] for t in tracks)
    if total >= MIN_SECONDS or n == 0:
        return tracks
    extra = min(MAX_EXTRA_PAUSE, (MIN_SECONDS - total) / n + 1 / FPS)
    gap = np.zeros(int(extra * SAMPLE_RATE), dtype=np.float32)
    for i in range(1, n + 1):
        audio, _, captions = tracks[i]
        tracks[i] = _pad_to_frames(np.concatenate([audio, gap])) + (captions,)
    return tracks


def make_video(
    story: Story,
    out_dir: str | Path = "outputs",
    voice: str = "Heart (warm female, US)",
    speed: float = 0.9,
    motion: bool = False,
    references=None,
    reference_strength: float = 0.6,
    motion_strength: float = 0.6,
    music: str | Path | None = None,
    music_volume: float = 0.12,
    seed: int = 42,
    progress: Progress | None = None,
) -> Path:
    progress = progress or (lambda f, msg: print(f"[{f:4.0%}] {msg}", flush=True))
    run_dir = Path(out_dir) / time.strftime("%Y%m%d-%H%M%S")
    (run_dir / "scenes").mkdir(parents=True, exist_ok=True)
    (run_dir / "story.json").write_text(story.to_json(), encoding="utf-8")
    n = len(story.scenes)

    release_writer()
    refs = load_references(references)
    if refs:
        (run_dir / "references").mkdir(exist_ok=True)
        for i, r in enumerate(refs):
            r.save(run_dir / "references" / f"ref_{i + 1}.png")
    progress(0.02, "Loading narrator voice")
    narrator = Narrator(voice, speed)
    tracks = []
    title_audio, _ = _narrate(narrator, story.title + ".")
    tracks.append(_pad_to_frames(title_audio) + ([],))
    for i, scene in enumerate(story.scenes):
        progress(0.05 + 0.15 * i / n, f"Narrating scene {i + 1}/{n}")
        audio, captions = _narrate(narrator, scene.narration)
        tracks.append(_pad_to_frames(audio) + (captions,))
    end_audio, end_caps = _narrate(narrator, "The end.")
    tracks.append(_pad_to_frames(end_audio) + (end_caps,))
    tracks = _stretch_to_minimum(tracks, n)
    del narrator

    from .images import Illustrator

    progress(0.2, "Loading illustrator")
    illustrator = Illustrator(refs, reference_strength)
    images = []
    for i, scene in enumerate(story.scenes):
        progress(0.22 + (0.33 if motion else 0.58) * i / n, f"Drawing scene {i + 1}/{n}")
        img = illustrator.draw(scene.picture or scene.narration, story.characters, story.setting, seed + i)
        img.save(run_dir / "scenes" / f"scene_{i + 1:02d}.png")
        images.append(img)
    illustrator.close()

    clips: list = [None] * n
    MOTION_FPS = 8
    if motion:
        from .motion import FPS as MOTION_FPS
        from .motion import Animator

        progress(0.55, "Loading animator")
        animator = Animator(refs, reference_strength)
        slow = "" if has_gpu() else " (a few minutes each on CPU)"
        for i, scene in enumerate(story.scenes):
            progress(0.57 + 0.23 * i / n, f"Animating scene {i + 1}/{n}{slow}")
            clips[i] = animator.animate(images[i], scene.picture or scene.narration, story.characters,
                                        story.setting, seed + i, strength=motion_strength)
            clips[i][0].save(run_dir / "scenes" / f"scene_{i + 1:02d}_anim_first.png")
        animator.close()

    shots = [Shot(images[0], tracks[0][1], title=story.title)]
    for i in range(n):
        shots.append(Shot(images[i], tracks[i + 1][1], tracks[i + 1][2], clip=clips[i],
                          clip_fps=MOTION_FPS if motion else 8))
    shots.append(Shot(images[-1], tracks[-1][1], title="The End"))

    audio_path = run_dir / "narration.wav"
    sf.write(audio_path, np.concatenate([t[0] for t in tracks]), SAMPLE_RATE)
    safe = re.sub(r"[^A-Za-z0-9]+", "_", story.title).strip("_")[:50] or "story"
    out_path = run_dir / f"{safe}.mp4"
    render(shots, audio_path, out_path, Path(music) if music else None, music_volume,
           progress=lambda f: progress(0.8 + 0.19 * f, "Putting the video together"))
    progress(1.0, "Done")
    return out_path


__all__ = ["make_video", "write_story", "story_from_text", "Story", "Scene"]
