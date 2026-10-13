# KidVid: a free, personal kids video maker

Turn a one-line idea into a narrated kids video that runs for several minutes, with pictures, gentle motion, subtitles, a title card, and optional background music. Everything runs on free, open models on your own machine or in free Google Colab. There are no paid APIs and no accounts to sign up for.

| Step | Model (free, open) | Runs on |
|---|---|---|
| Story script | Qwen2.5 Instruct (1.5B on CPU, 3B on GPU) | CPU or GPU |
| Pictures | SD-Turbo (CPU) / SDXL-Turbo (GPU) | CPU or GPU |
| Narration | Kokoro-82M | CPU or GPU |
| Optional AI motion | LTX-Video (image-to-video) | GPU only (Colab T4 is fine) |
| Video assembly | ffmpeg (Ken Burns pan/zoom, captions, fades, music mix) | CPU |

Video length depends on how many scenes you ask for. Each scene lasts about 10-15 seconds, so 12 scenes come to roughly 2.5 minutes and 24 scenes to roughly 5 minutes.

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

1. **Write the story.** Type an idea and pick the number of scenes, or paste your own story with a blank line between scenes.
2. **Edit the script** (JSON) if you like. `characters` is added to every picture prompt to keep characters consistent.
3. Pick a voice and speed, optionally add a music file, and click **Make video**.

Each run is saved to `outputs/<timestamp>/` and contains the MP4, `story.json`, the narration WAV, and every scene picture.

Command-line use:

```python
from kidvid.story import write_story
from kidvid.pipeline import make_video
story = write_story("a brave little bee who is scared of flowers", num_scenes=16)
print(make_video(story, voice="Heart (warm female, US)", motion=False))
```

## Notes

- Model licences: SD-Turbo/SDXL-Turbo are licensed for non-commercial use, which is fine for personal use. Do not sell videos made with them without checking each licence.
- Free Colab sessions time out after a few hours, and the models download again each session.
- AI motion takes about 1-3 minutes per scene on a T4. Storybook mode is much faster.
