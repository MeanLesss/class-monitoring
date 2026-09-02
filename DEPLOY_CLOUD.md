# Deploy to Streamlit Community Cloud (free)

This folder is set up to deploy to **Streamlit Community Cloud** (free tier:
~1 GB RAM, 1 CPU, no GPU). The app used for the cloud is **`cloud_app.py`** —
it's the same YOLO detection pipeline as `app.py`, but without the live WebRTC
stream (which needs a paid TURN relay and is unreliable on the free cloud).

On the cloud you get:
- **On-demand photo capture** (`st.camera_input`) — click, allow camera, press
  Run detection. No live streaming.
- **Image upload** — draw seat boxes, detect heads inside them.
- **Video upload** — short clips, per-frame counting.
- **Automatic model download** — the pretrained YOLO weights aren't stored in
  git; `cloud_app.py` downloads them from Ultralytics on first run.

## What changed for the cloud

| File | Purpose |
|------|---------|
| `cloud_app.py` | Cloud entry point (no WebRTC; uses on-demand capture + uploads) |
| `requirements.txt` | Cloud-safe deps (opencv-headless, no `streamlit-webrtc`/`av`) |
| `.streamlit/config.toml` | Cloud-friendly Streamlit config |

`app.py` (local WebRTC live stream) is unchanged. To run it locally you need the
extra deps: `pip install streamlit-webrtc av`.

## Step-by-step deploy

1. **Make sure your GitHub repo has the files.** The repo is:
   `https://github.com/MeanLesss/class-monitoring.git`. Push `cloud_app.py`,
   `requirements.txt`, `.streamlit/config.toml`, and the `seatdraw/` folder.
   (The `*.pt` weights are intentionally not committed — they download at
   runtime. `ai-venv/`, `runs/`, `dataset/` are gitignored.)

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
  CPU.
- The default person model is `yolo11n-pose` (small, fast). On uploads you can
  switch models in the sidebar, but avoid heavy ones to stay under RAM.
- First load downloads weights + warms up models, so the first request is slow;
  later ones are faster.
- If the app shows "Error: No module named ...", confirm `requirements.txt` is
  at the **repo root** (Community Cloud only reads that exact filename).

## Local run (this machine)

```powershell
.\ai-venv\Scripts\streamlit run cloud_app.py
```
