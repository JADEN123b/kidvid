# KidVid: a free, personal kids video maker

Turn a one-line idea into a narrated, animated kids video that runs for several minutes. You can chat with a story assistant, upload reference pictures so every scene matches your character or art style, and add optional background music. Everything runs on free, open models on your own machine or in free Google Colab. There are no paid APIs and no accounts to sign up for.

| Step | Model (free, open) | Runs on |
|---|---|---|
| Story script + chatbot | Qwen2.5 Instruct (1.5B on CPU, 3B on GPU) | CPU or GPU |
| Pictures | DreamShaper 8 (SD 1.5) + LCM-LoRA, 4-6 steps | CPU or GPU |
| Match reference pictures | IP-Adapter (SD 1.5) | CPU or GPU |
| Animation | AnimateDiff + AnimateLCM, picture-to-animation (video-to-video) | GPU fast; CPU a few minutes per scene |
| Narration | Kokoro-82M | CPU or GPU |
| Video assembly | ffmpeg (Ken Burns pan/zoom, captions, fades, music mix) | CPU |

Video length depends on how many scenes you ask for (12-30, default 14). A full video always lasts at least 1.5 minutes: if the narration is short, each scene is held a little longer. 14 scenes come to roughly 1.5-2.5 minutes and 30 scenes to roughly 5 minutes.

## Option A: Google Colab (free GPU, any computer)

1. Open `KidVid_Colab.ipynb` in Colab (on GitHub, open the file and click "Open in Colab", or upload it at colab.research.google.com).
2. Go to **Runtime > Change runtime type > T4 GPU**.
3. Set a password, then go to **Runtime > Run all**.
4. Open the `gradio.live` link it prints and log in with user `me` and your password.

## Option B: your own computer

Requirements: Python 3.10+, ffmpeg, and espeak-ng (`sudo apt install ffmpeg espeak-ng`, `brew install ffmpeg espeak-ng`, or install both on Windows and add them to PATH). A GPU is optional. Without one, storybook mode still works, just more slowly. 16 GB RAM is recommended.

```bash
git clone https://github.com/JADEN123b/kidvid.git && cd kidvid
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt                      # NVIDIA GPU: install CUDA torch first (pytorch.org)
python app.py                                        # then open http://localhost:7860
```

## Using it

1. **Write the story.** Pick one:
   - **Chat with Buddy:** for example "write a story about a dragon who loves baking, 16 scenes", then "make scene 3 funnier" or "add a talking owl after scene 2".
   - **Quick story:** type an idea and pick the number of scenes.
   - **My own story:** paste your text with a blank line between scenes.
2. **Edit the script** (JSON) if you like. `characters` is added to every picture prompt to keep characters consistent.
3. **Optional reference pictures:** upload your character or an art style. The slider controls how closely scenes copy it.
4. **Animate:** leave it ticked for moving scenes, or untick it for faster storybook-style pan and zoom.
5. Pick a voice, optionally add music, and click **Make video**.

Each run is saved to `outputs/<timestamp>/` and contains the MP4, `story.json`, the narration WAV, and every scene picture.

Command-line use:

```python
from kidvid.story import write_story
from kidvid.pipeline import make_video
story = write_story("a brave little bee who is scared of flowers", num_scenes=16)
print(make_video(story, voice="Heart (warm female, US)", motion=False))
```

## Notes

- Every model here is free to download. Check each model's licence before using videos commercially.
- Free Colab sessions time out after a few hours, and the models download again each session.
- Animation takes seconds per scene on a Colab T4 and a few minutes per scene on a CPU. Untick it for fast storybook mode.
- Free Colab: run the notebook with the T4 GPU runtime for the best speed and the bigger story model.
