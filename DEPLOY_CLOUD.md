# Deploy to Streamlit Community Cloud (free)

This folder is set up to deploy to **Streamlit Community Cloud** (free tier:
~1 GB RAM, 1 CPU, no GPU). The app used for the cloud is **`cloud_app.py`** — a
live camera feed plus cloud-friendly upload/photo inputs, and your trained model.

On the cloud you get:
- **Live camera feed** — via `streamlit-camera-input-live`, which streams your
  browser webcam as frames WITHOUT WebRTC (so no TURN relay is needed and it
  works reliably on the free cloud). A few frames/sec on CPU.
- **On-demand photo capture** (`st.camera_input`) — always works, no live stream.
- **Image upload** — draw seat boxes, detect heads inside them.
- **Video upload** — short clips, per-frame counting.
- **Your trained model committed** — `models/tuned/class-monitor-ai.pt` is in the
  repo, selectable in the sidebar as "class-monitor-ai (my trained model)".
- **Automatic pretrained model download** — the pretrained YOLO weights aren't in
  git; `cloud_app.py` downloads them from Ultralytics on first run.

## What changed for the cloud

| File | Purpose |
|------|---------|
| `cloud_app.py` | Cloud entry point (live camera + on-demand capture + uploads) |
| `requirements.txt` | Cloud-safe deps (opencv-headless + camera-input-live for live feed) |
| `models/tuned/class-monitor-ai.pt` | Your trained model, committed so it runs on the cloud |
| `.streamlit/config.toml` | Cloud-friendly Streamlit config |

`app.py` (local WebRTC live stream) remains for local use only — WebRTC does not
work reliably on the free Streamlit Cloud (needs a paid TURN relay), so the cloud
app uses `camera-input-live` instead.

## Step-by-step deploy

1. **Make sure your GitHub repo has the files.** The repo is:
   `https://github.com/MeanLesss/class-monitoring.git`. Push `cloud_app.py`,
   `requirements.txt`, `.streamlit/config.toml`, `models/tuned/class-monitor-ai.pt`
   and the `seatdraw/` folder. (The pretrained `*.pt` weights are not in git —
   they download at runtime. `ai-venv/`, `runs/`, `dataset/` are gitignored.)

2. **Go to** https://share.streamlit.io/deploy (sign in with GitHub).

3. Click **New app** → connect your `class-monitoring` repo.

4. Set:
   - **Repository**: `MeanLesss/class-monitoring`
   - **Branch**: `main`
   - **Main file path**: `classroom_occupancy/cloud_app.py`
     (adjust to the actual path if the repo root is `classroom_occupancy/`)
   - **App URL**: pick a name.

5. Click **Deploy**.

6. Wait for the build (first run downloads the YOLO weights + installs deps, a
   few minutes). You should see the app with the camera / upload / video tabs.

> Note: the repo root for the git remote is the `classroom_occupancy` folder
> (the repo was initialised there), so `main file path` is `cloud_app.py` if the
> app lives at repo root, otherwise `classroom_occupancy/cloud_app.py`.

## Tips / limits (free tier)

- Keep uploaded **videos short** (≤ ~30 s) — each frame runs two YOLO models on
  CPU. "Build annotated video" runs detection once and saves a playable H.264
  MP4 you can play / pause / seek (uses PyAV; needs `av` in requirements.txt).
- The default person model is `yolo11n-pose` (small, fast). On uploads you can
  switch models in the sidebar (including your trained `class-monitor-ai`).
- The **live camera** runs YOLO on every frame, so it updates at a low FPS on
  the free cloud (CPU-only). Tick "Start live detection" and allow camera access;
  untick to stop. For a faster single check, use the on-demand photo capture.
- First load downloads weights + warms up models, so the first request is slow;
  later ones are faster.
- If the app shows "Error: No module named ...", confirm `requirements.txt` is
  at the **repo root** (Community Cloud only reads that exact filename).

## Local run (this machine)

```powershell
.\ai-venv\Scripts\streamlit run cloud_app.py
```
