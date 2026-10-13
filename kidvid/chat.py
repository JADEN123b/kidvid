import json
import re
from threading import Thread

import torch
from transformers import TextIteratorStreamer

from .device import device
from .story import Scene, Story, _extract_json, _flatten, get_writer

SYSTEM = """You are KidVid Buddy, a cheerful helper for making children's story videos (kids aged 3-8).
You brainstorm ideas, answer questions and change the story script. Keep everything kind and kid-friendly.

Current script:
{story}

If the user wants to CHANGE the script: reply in one or two friendly sentences, then ONE ```json block using only the keys you need:
{{"title": "...", "characters": "...", "setting": "...", "edit": [{{"scene": 3, "narration": "...", "picture": "..."}}], "delete": [5], "add": [{{"after": 2, "narration": "...", "picture": "..."}}]}}
Scene numbers start at 1. "narration" is 2-3 short sentences read aloud. "picture" describes one illustration in under 20 words.

If the user wants a BRAND NEW story: reply briefly, then:
```json
{{"new_story": {{"idea": "...", "scenes": 12}}}}
```

Otherwise just chat, with no JSON."""


def _script_for_prompt(story: Story | None) -> str:
    if story is None or not story.scenes:
        return "(empty - no story yet)"
    lines = [f"Title: {story.title}", f"Characters: {story.characters}", f"Setting: {story.setting}"]
    for i, s in enumerate(story.scenes, 1):
        lines.append(f"Scene {i}: {s.narration} [picture: {s.picture}]")
    return "\n".join(lines)


def stream_reply(history: list[dict], story: Story | None, max_turns: int = 8):
    """Yield the assistant's reply text as it is generated."""
    writer = get_writer()
    messages = [{"role": "system", "content": SYSTEM.format(story=_script_for_prompt(story))}]
    messages += [{"role": m["role"], "content": _text(m["content"])} for m in history[-max_turns:]]
    inputs = writer.tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(device())
    streamer = TextIteratorStreamer(writer.tok, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(**inputs, streamer=streamer, max_new_tokens=700, do_sample=True, temperature=0.7, top_p=0.9,
                  pad_token_id=writer.tok.eos_token_id)
    thread = Thread(target=lambda: torch.no_grad()(writer.model.generate)(**kwargs))
    thread.start()
    text = ""
    for piece in streamer:
        text += piece
        yield text
    thread.join()


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    return str(content)


def split_reply(text: str) -> tuple[str, dict | None]:
    """Separate the friendly chat text from the JSON command block (if any)."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if not match:
        start = text.find("{")
        if start == -1:
            return text.strip(), None
        try:
            return text[:start].strip(), _extract_json(text[start:])
        except (ValueError, json.JSONDecodeError):
            return text.strip(), None
    try:
        data = _extract_json(match.group(1))
    except (ValueError, json.JSONDecodeError):
        return text.strip(), None
    return (text[: match.start()] + text[match.end() :]).strip(), data


def apply_changes(story: Story | None, data: dict) -> tuple[Story, list[str]]:
    story = story or Story(title="My Story", characters="", setting="", scenes=[])
    if isinstance(data.get("scenes"), list) and data["scenes"]:
        return Story.from_json(json.dumps(data)), ["replaced the whole script"]
    notes = []
    for key in ("title", "characters", "setting"):
        if _flatten(data.get(key, "")):
            setattr(story, key, _flatten(data[key]))
            notes.append(f"updated {key}")
    for e in data.get("edit", []) or []:
        try:
            idx = int(e.get("scene", 0)) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(story.scenes):
            if _flatten(e.get("narration", "")):
                story.scenes[idx].narration = _flatten(e["narration"])
            if _flatten(e.get("picture", "")):
                story.scenes[idx].picture = _flatten(e["picture"])
            notes.append(f"edited scene {idx + 1}")
        elif _flatten(e.get("narration", "")):
            story.scenes.append(Scene(_flatten(e["narration"]), _flatten(e.get("picture", "")) or _flatten(e["narration"])))
            notes.append(f"added scene {len(story.scenes)}")
    deletes = sorted({int(d) for d in (data.get("delete", []) or []) if str(d).isdigit()}, reverse=True)
    for d in deletes:
        if 1 <= d <= len(story.scenes):
            story.scenes.pop(d - 1)
            notes.append(f"deleted scene {d}")
    for a in sorted(data.get("add", []) or [], key=lambda a: -int(a.get("after", 0) or 0)):
        narration = _flatten(a.get("narration", ""))
        if not narration:
            continue
        after = max(0, min(int(a.get("after", len(story.scenes)) or 0), len(story.scenes)))
        story.scenes.insert(after, Scene(narration, _flatten(a.get("picture", "")) or narration))
        notes.append(f"added a scene after scene {after}")
    return story, notes
