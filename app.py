import argparse
import os

import gradio as gr

from kidvid.chat import apply_changes, split_reply, stream_reply
from kidvid.device import has_gpu
from kidvid.pipeline import make_video, story_from_text
from kidvid.story import Story, write_story
from kidvid.tts import VOICES

GPU = has_gpu()


def _current_story(story_json):
    try:
        return Story.from_json(story_json, allow_empty=True) if (story_json or "").strip() else None
    except Exception:
        return None


def on_write_story(idea, num_scenes, progress=gr.Progress()):
    if not idea.strip():
        raise gr.Error("Type a story idea first.")
    progress(0.1, desc="Writing the story (this can take a few minutes on CPU)")
    return write_story(idea, int(num_scenes)).to_json()


def on_use_own_text(text, title):
    if not text.strip():
        raise gr.Error("Paste your story text first.")
    return story_from_text(text, title=title.strip() or "My Story").to_json()


def on_chat(message, history, story_json):
    message = (message or "").strip()
    if not message:
        yield history, story_json, ""
        return
    history = (history or []) + [{"role": "user", "content": message}]
    story = _current_story(story_json)
    shown = history + [{"role": "assistant", "content": "..."}]
    yield shown, story_json, ""
    reply = ""
    for reply in stream_reply(history, story):
        visible = reply.split("```")[0].strip() or "..."
        shown[-1] = {"role": "assistant", "content": visible + (" *(updating the script...)*" if "```" in reply else "")}
        yield shown, story_json, ""
    text, command = split_reply(reply)
    text = text or "Okay!"
    if command and isinstance(command.get("new_story"), dict):
        spec = command["new_story"]
        idea = str(spec.get("idea") or message)
        try:
            scenes = max(12, min(30, int(spec.get("scenes", 14))))
        except (TypeError, ValueError):
            scenes = 14
        shown[-1] = {"role": "assistant", "content": f"{text}\n\n*Writing a {scenes}-scene story about: {idea} ...*"}
        yield shown, story_json, ""
        new = write_story(idea, scenes)
        shown[-1] = {"role": "assistant",
                     "content": f"{text}\n\n**Done!** \"{new.title}\" with {len(new.scenes)} scenes is in the script box."}
        yield shown, new.to_json(), ""
        return
    if command:
        try:
            story, notes = apply_changes(story, command)
            story_json = story.to_json()
            text += "\n\n*Script updated: " + (", ".join(notes) or "no changes") + ".*"
        except Exception as e:
            text += f"\n\n*(I tried to change the script but couldn't: {e})*"
    shown[-1] = {"role": "assistant", "content": text}
    yield shown, story_json, ""


def on_make_video(story_json, voice, speed, references, ref_strength, motion, motion_strength, music,
                  music_volume, seed, progress=gr.Progress()):
    if not (story_json or "").strip():
        raise gr.Error("No story yet. Do step 1 first: chat, write a story or paste your own, then click Make video.")
    try:
        story = Story.from_json(story_json)
    except Exception as e:
        raise gr.Error(f"The story script is not valid: {e}")
    path = make_video(
        story,
        voice=voice,
        speed=speed,
        motion=motion,
        references=references,
        reference_strength=ref_strength,
        motion_strength=motion_strength,
        music=music,
        music_volume=music_volume,
        seed=int(seed),
        progress=lambda f, msg: progress(f, desc=msg),
    )
    scenes = sorted((path.parent / "scenes").glob("scene_??.png"))
    return str(path), [str(p) for p in scenes]


with gr.Blocks(title="KidVid - personal kids video maker") as demo:
    gr.Markdown(
        "# KidVid\nMake a full narrated, animated kids video from an idea. "
        f"Running on **{'GPU' if GPU else 'CPU (slow but works)'}**."
    )
    with gr.Row():
        with gr.Column():
            with gr.Tab("1a. Chat with Buddy"):
                chatbot = gr.Chatbot(label="KidVid Buddy", height=380)
                chat_in = gr.Textbox(
                    placeholder="e.g. Write a story about a dragon who loves baking, 10 scenes / Make scene 2 funnier",
                    show_label=False,
                )
                chat_btn = gr.Button("Send", variant="primary")
            with gr.Tab("1b. Quick story"):
                idea = gr.Textbox(label="Story idea", placeholder="A shy little turtle who learns to swim with new friends")
                num_scenes = gr.Slider(12, 30, value=14, step=1, label="Number of scenes (videos are always at least 1.5 minutes)")
                write_btn = gr.Button("Write story", variant="primary")
            with gr.Tab("1c. My own story"):
                own_title = gr.Textbox(label="Title", value="My Story")
                own_text = gr.Textbox(label="Story text (blank line between scenes)", lines=10)
                own_btn = gr.Button("Use this text")
            story_box = gr.Code(label="2. Story script (edit anything you like)", language="json", lines=18)
        with gr.Column():
            references = gr.File(
                label="Reference pictures (optional): your character or an art style to copy",
                file_count="multiple",
                file_types=["image"],
            )
            ref_strength = gr.Slider(0.0, 1.0, value=0.6, step=0.05,
                                     label="How closely to follow the reference pictures")
            motion = gr.Checkbox(
                label="Animate each scene (AnimateDiff)" + ("" if GPU else " - slow on CPU, a few minutes per scene"),
                value=GPU,
            )
            motion_strength = gr.Slider(0.3, 0.9, value=0.6, step=0.05,
                                        label="Animation amount (higher = more movement, less like the picture)")
            voice = gr.Dropdown(list(VOICES), value=list(VOICES)[0], label="Narrator voice")
            speed = gr.Slider(0.7, 1.2, value=0.9, step=0.05, label="Narration speed")
            music = gr.Audio(label="Background music (optional)", type="filepath")
            music_volume = gr.Slider(0.0, 0.5, value=0.12, step=0.01, label="Music volume")
            seed = gr.Number(value=42, precision=0, label="Seed (change for different pictures)")
            make_btn = gr.Button("3. Make video", variant="primary")
            video = gr.Video(label="Your video")
            gallery = gr.Gallery(label="Scene pictures", columns=4, height="auto")

    chat_inputs = [chat_in, chatbot, story_box]
    chat_outputs = [chatbot, story_box, chat_in]
    chat_btn.click(on_chat, chat_inputs, chat_outputs)
    chat_in.submit(on_chat, chat_inputs, chat_outputs)
    write_btn.click(on_write_story, [idea, num_scenes], story_box)
    own_btn.click(on_use_own_text, [own_text, own_title], story_box)
    make_btn.click(
        on_make_video,
        [story_box, voice, speed, references, ref_strength, motion, motion_strength, music, music_volume, seed],
        [video, gallery],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create a public gradio.live link (use with --password)")
    parser.add_argument("--password", default=os.environ.get("KIDVID_PASSWORD"), help="Protect the app with a password")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    auth = ("me", args.password) if args.password else None
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0", server_port=args.port, share=args.share, auth=auth, allowed_paths=["outputs"]
    )
