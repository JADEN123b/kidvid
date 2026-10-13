import json
import re
from dataclasses import asdict, dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .device import device, free_memory, has_gpu

CPU_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
GPU_MODEL = "Qwen/Qwen2.5-3B-Instruct"

SYSTEM = (
    "You are a children's author who writes gentle, fun, educational stories for kids aged 3-8. "
    "You always answer with valid JSON only, no markdown."
)

OUTLINE_PROMPT = """Plan a children's story video about: {idea}

Give exactly {n} story beats. Each beat is one short sentence saying what happens next.
The story has a clear beginning, a middle with a small problem, and a happy ending with a simple lesson.
"characters" describes the main characters' look in one short line (species, colours, clothes).
"setting" is one short line describing where the story happens.

Answer with JSON in exactly this shape:
{{"title": "...", "characters": "...", "setting": "...", "beats": ["...", "..."]}}"""

EXPAND_PROMPT = """Here is the plan for a children's story called "{title}":
{plan}

This plan has {have} beats but needs exactly {n}. Rewrite it as exactly {n} beats by adding small, fun in-between moments.
Keep the same beginning and happy ending. Each beat is one short sentence.

Answer with JSON in exactly this shape:
{{"beats": ["...", "..."]}}"""

SCENES_PROMPT = """We are writing a children's story video called "{title}".
Characters: {characters}
Setting: {setting}
Full plan:
{plan}

Now write scenes {first} to {last}, one scene per beat:
{beats}

For each scene:
- "narration": exactly THREE short, simple, fun sentences (about 30-40 words in total) read aloud to kids aged 3-8.
  Example narration: "Benny the bear woke up with a big yawn. The sun was shining and the birds were singing. Today was the day of the great picnic!"
- "picture": ONE illustration in under 20 words: who is in it, what they are doing, where. No text in the picture.

Answer with JSON in exactly this shape:
{{"scenes": [{{"narration": "...", "picture": "..."}}]}}"""


@dataclass
class Scene:
    narration: str
    picture: str


@dataclass
class Story:
    title: str
    characters: str
    setting: str
    scenes: list[Scene] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str, allow_empty: bool = False) -> "Story":
        data = _extract_json(text)
        scenes = [
            Scene(narration=str(s.get("narration", "")).strip(), picture=str(s.get("picture", "")).strip())
            for s in data.get("scenes", [])
            if str(s.get("narration", "")).strip()
        ]
        if not scenes and not allow_empty:
            raise ValueError("Story has no scenes with narration.")
        return cls(
            title=str(data.get("title", "My Story")).strip(),
            characters=_flatten(data.get("characters", "")),
            setting=_flatten(data.get("setting", "")),
            scenes=scenes,
        )


def _flatten(value) -> str:
    if isinstance(value, dict):
        return "; ".join(f"{k}: {_flatten(v)}" for k, v in value.items())
    if isinstance(value, list):
        return "; ".join(_flatten(v) for v in value)
    return str(value).strip()


def _extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in story text.")
    blob = text[start : end + 1]
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        blob = re.sub(r",\s*([}\]])", r"\1", blob)
        return json.loads(blob)


class _Writer:
    def __init__(self, model_id: str):
        dtype = torch.float16 if has_gpu() else torch.bfloat16
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device())

    def ask_json(self, prompt: str, max_new_tokens: int, attempts: int = 3) -> dict:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        inputs = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(device())
        prompt_len = inputs["input_ids"].shape[1]
        last_error: Exception | None = None
        for _ in range(attempts):
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=self.tok.eos_token_id,
                )
            try:
                return _extract_json(self.tok.decode(out[0][prompt_len:], skip_special_tokens=True))
            except (ValueError, json.JSONDecodeError) as e:
                last_error = e
        raise RuntimeError(f"The story model did not return valid JSON: {last_error}")

    def close(self) -> None:
        del self.model
        free_memory()


_cached: _Writer | None = None


def get_writer() -> _Writer:
    """The story model stays loaded between chat messages; call release_writer() before heavy image work."""
    global _cached
    if _cached is None:
        _cached = _Writer(GPU_MODEL if has_gpu() else CPU_MODEL)
    return _cached


def release_writer() -> None:
    global _cached
    if _cached is not None:
        _cached.close()
        _cached = None


def write_story(idea: str, num_scenes: int = 12, batch: int = 4) -> Story:
    """Plan the story as beats first, then write the scenes a few beats at a time (keeps small models on track)."""
    writer = get_writer()
    outline = writer.ask_json(OUTLINE_PROMPT.format(idea=idea.strip(), n=num_scenes), 40 * num_scenes + 200)
    beats = [_flatten(b) for b in outline.get("beats", []) if _flatten(b)][:num_scenes]
    if not beats:
        raise RuntimeError("The story model returned no story beats.")
    title = _flatten(outline.get("title", "")) or "My Story"
    if len(beats) < num_scenes:
        plan = "\n".join(f"{i + 1}. {b}" for i, b in enumerate(beats))
        try:
            more = writer.ask_json(
                EXPAND_PROMPT.format(title=title, plan=plan, have=len(beats), n=num_scenes), 40 * num_scenes + 100
            )
            expanded = [_flatten(b) for b in more.get("beats", []) if _flatten(b)][:num_scenes]
            if len(expanded) > len(beats):
                beats = expanded
        except RuntimeError:
            pass
    characters = _flatten(outline.get("characters", ""))
    setting = _flatten(outline.get("setting", ""))
    plan = "\n".join(f"{i + 1}. {b}" for i, b in enumerate(beats))
    scenes: list[Scene] = []
    for first in range(0, len(beats), batch):
        chunk = beats[first : first + batch]
        prompt = SCENES_PROMPT.format(
            title=title, characters=characters, setting=setting, plan=plan,
            first=first + 1, last=first + len(chunk),
            beats="\n".join(f"{first + i + 1}. {b}" for i, b in enumerate(chunk)),
        )
        data = writer.ask_json(prompt, 110 * len(chunk) + 100)
        written = [s for s in data.get("scenes", []) if isinstance(s, dict)]
        for i, beat in enumerate(chunk):
            s = written[i] if i < len(written) else {}
            narration = _flatten(s.get("narration", "")) or beat
            scenes.append(Scene(narration=narration, picture=_flatten(s.get("picture", "")) or beat))
    return Story(title=title, characters=characters, setting=setting, scenes=scenes)
