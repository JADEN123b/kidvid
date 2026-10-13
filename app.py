import argparse
import os

import gradio as gr

from kidvid.device import has_gpu
from kidvid.pipeline import make_video, story_from_text
from kidvid.story import Story, write_story
from kidvid.tts import VOICES

GPU = has_gpu()


def on_write_story(idea, num_scenes, progress=gr.Progress()):
    if not idea.strip():
        raise gr.Error("Type a story idea first.")
    progress(0.1, desc="Writing the story (this can take a few minutes on CPU)")
    return write_story(idea, int(num_scenes)).to_json()


def on_use_own_text(text, title):
    if not text.strip():
        raise gr.Error("Paste your story text first.")
    return story_from_text(text, title=title.strip() or "My Story").to_json()


def on_make_video(story_json, voice, speed, motion, music, music_volume, seed, progress=gr.Progress()):
    if not (story_json or "").strip():
        raise gr.Error("No story yet. Do step 1 first: write a story or paste your own, then click Make video.")
    try:
        story = Story.from_json(story_json)
    except Exception as e:
        raise gr.Error(f"The story JSON is not valid: {e}")
    if motion and not GPU:
        raise gr.Error("AI motion needs a GPU. Untick it, or run this in Google Colab.")
    path = make_video(
        story,
        voice=voice,
        speed=speed,
        motion=motion,
        music=music,
        music_volume=music_volume,
        seed=int(seed),
        progress=lambda f, msg: progress(f, desc=msg),
    )
    scenes = sorted((path.parent / "scenes").glob("*.png"))
    return str(path), [str(p) for p in scenes]


with gr.Blocks(title="KidVid - personal kids video maker") as demo:
    gr.Markdown(
        "# KidVid\nMake a full narrated kids video from an idea. "
        f"Running on **{'GPU' if GPU else 'CPU'}**."
    )
    with gr.Row():
        with gr.Column():
            with gr.Tab("1a. Let AI write the story"):
                idea = gr.Textbox(label="Story idea", placeholder="A shy little turtle who learns to swim with new friends")
                num_scenes = gr.Slider(4, 30, value=12, step=1, label="Number of scenes (~10-15 seconds each)")
                write_btn = gr.Button("Write story", variant="primary")
            with gr.Tab("1b. Use my own story"):
                own_title = gr.Textbox(label="Title", value="My Story")
                own_text = gr.Textbox(label="Story text (blank line between scenes)", lines=10)
                own_btn = gr.Button("Use this text")
            story_box = gr.Code(label="2. Story script (edit anything you like)", language="json", lines=20)
        with gr.Column():
            voice = gr.Dropdown(list(VOICES), value=list(VOICES)[0], label="Narrator voice")
            speed = gr.Slider(0.7, 1.2, value=0.9, step=0.05, label="Narration speed")
            motion = gr.Checkbox(
                label="AI motion (animate each scene with LTX-Video; GPU only, ~1-3 min per scene)",
                value=False,
                interactive=GPU,
            )
            music = gr.Audio(label="Background music (optional)", type="filepath")
            music_volume = gr.Slider(0.0, 0.5, value=0.12, step=0.01, label="Music volume")
            seed = gr.Number(value=42, precision=0, label="Seed (change for different pictures)")
            make_btn = gr.Button("3. Make video", variant="primary")
            video = gr.Video(label="Your video")
            gallery = gr.Gallery(label="Scene pictures", columns=4, height="auto")

    write_btn.click(on_write_story, [idea, num_scenes], story_box)
    own_btn.click(on_use_own_text, [own_text, own_title], story_box)
    make_btn.click(on_make_video, [story_box, voice, speed, motion, music, music_volume, seed], [video, gallery])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create a public gradio.live link (use with --password)")
    parser.add_argument("--password", default=os.environ.get("KIDVID_PASSWORD"), help="Protect the app with a password")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    auth = ("me", args.password) if args.password else None
    demo.queue().launch(server_name="0.0.0.0", server_port=args.port, share=args.share, auth=auth,
                        allowed_paths=["outputs"])
