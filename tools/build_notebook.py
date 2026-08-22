"""Generates verify_yolo.ipynb (valid JSON) from cell definitions.

Run: python tools/build_notebook.py
"""

import json
import os

CELLS = [
    # 1 ---- title
    ("md", """# Classroom Occupancy — Verify + Collect + Fine-tune

One notebook for the whole loop: **verify** pretrained YOLO on your classroom
angles, **add photos** via webcam or an image picker, then **learn from your
image dataset** by fine-tuning.

What you can do here:
1. **Webcam** — requests the camera automatically (local camera first, then the
   browser's). If no camera is found, an **image picker** appears instead.
2. **Image picker** — pick/upload photos (e.g. taken on your phone); they land in
   `dataset/images/`.
3. **Verify** — run pretrained YOLOv8 / YOLO11 (detect + pose) over every photo
   and compare counts/confidences at different classroom angles.
4. **Train (image loop)** — auto-label every unlabeled photo, split train/val,
   then fine-tune `yolo11n.pt` so the model *learns your seats and angles*.

> No camera on this PC? Use the **image picker** in section 1, or run
> `app.py` (Streamlit) and capture from your phone.
"""),

    # 2 ---- setup + imports
    ("code", """import os
import time
import glob
import shutil
import warnings

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from base64 import b64decode
from IPython.display import display, Javascript
from IPython import get_ipython

from ultralytics import YOLO

warnings.filterwarnings("ignore")

DATA_DIR    = "dataset"
IMG_DIR     = os.path.join(DATA_DIR, "images")
LABEL_DIR   = os.path.join(DATA_DIR, "labels")
VAL_IMG_DIR = os.path.join(DATA_DIR, "val", "images")
VAL_LAB_DIR = os.path.join(DATA_DIR, "val", "labels")
OUT_DIR     = os.path.join(DATA_DIR, "outputs")
MODELS_DIR  = "models"

for d in (IMG_DIR, LABEL_DIR, VAL_IMG_DIR, VAL_LAB_DIR, OUT_DIR, MODELS_DIR):
    os.makedirs(d, exist_ok=True)

CONF = 0.30        # detection confidence threshold
PERSON_ID = 0      # COCO class id for "person"
CHAIR_ID = 56      # COCO class id for "chair"

# Pretrained weights to compare on local classroom angles
PRETRAINED = {
    "yolov8n":      "models/pretrained/yolov8n.pt",
    "yolo11n":      "models/pretrained/yolo11n.pt",
    "yolov8n-pose": "models/pretrained/yolov8n-pose.pt",
    "yolo11n-pose": "models/pretrained/yolo11n-pose.pt",
}


def list_images():
    return sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg"))
                  + glob.glob(os.path.join(IMG_DIR, "*.jpeg"))
                  + glob.glob(os.path.join(IMG_DIR, "*.png")))

def list_visual_images():
    \"\"\"Images to LOOK at: labeled set first, else plain screenshots
    (unlabeled -> visual checks only, NEVER training).\"\"\"
    labeled = list_images()
    if labeled:
        return labeled, f"labeled ({IMG_DIR})"
    shots = sorted(glob.glob(os.path.join("dataset", "plain_screenshots", "*.jpg")))
    return shots, "plain screenshots (UNLABELED - visual check only)"


images = list_images()
print(f"Found {len(images)} classroom photo(s) in {IMG_DIR}")
print("Plain app screenshots live in dataset/plain_screenshots - "
      "list_visual_images() falls back to them for visual-only cells.")
"""),

    # 3 ---- device check
    ("code", """import torch

device = "0" if torch.cuda.is_available() else "cpu"
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
print("Training device:", device, "| GPU:", gpu_name)
if not torch.cuda.is_available():
    print("WARNING: torch has no CUDA. Reinstall it with:  "
          "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130")
"""),

    # 3 ---- webcam capture helpers
    ("code", """def _decode_dataurl(dataurl):
    \"\"\"Convert a data:image/jpeg;base64,... string into a BGR frame.\"\"\"
    if not dataurl:
        return None, "no capture returned"
    b64 = dataurl.split(",")[1]
    arr = np.frombuffer(b64decode(b64), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR), None


def capture_local_webcam():
    \"\"\"Try the machine's own camera (OpenCV). Returns (frame, error).\"\"\"
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None, "no local camera detected"
    ok, frame = cap.read()
    cap.release()
    return (frame, None) if ok else (None, "local camera read failed")


def capture_browser_webcam():
    \"\"\"Request the BROWSER's camera (Colab / classic Jupyter). Returns (frame, error).\"\"\"
    js = \"\"\"
    new Promise((resolve, reject) => {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        reject('getUserMedia not available'); return;
      }
      navigator.mediaDevices.getUserMedia({video: true}).then(function(stream) {
        var v = document.createElement('video');
        v.srcObject = stream; v.muted = true; v.playsInline = true;
        v.style.cssText = 'max-width:320px;border:1px solid #ccc';
        document.body.appendChild(v);
        v.onloadedmetadata = function() {
          v.play();
          setTimeout(function() {
            var c = document.createElement('canvas');
            c.width = v.videoWidth || 640; c.height = v.videoHeight || 480;
            c.getContext('2d').drawImage(v, 0, 0);
            var data = c.toDataURL('image/jpeg', 0.85);
            stream.getTracks().forEach(function(t){ t.stop(); });
            resolve(data);
          }, 700);
        };
      }).catch(function(err){ reject(String(err)); });
    })
    \"\"\"
    # 1) Colab: eval_js returns the base64 string directly
    try:
        from google.colab.output import eval_js
        return _decode_dataurl(eval_js(js))
    except ImportError:
        pass

    # 2) Classic Jupyter: bridge JS -> kernel variable _CAM_IMG
    ip = get_ipython()
    if ip is not None and ip.kernel is not None:
        display(Javascript(js.replace(
            "resolve(data);",
            "resolve(data); if (window.IPython) { IPython.notebook.kernel.execute('_CAM_IMG = \"' + data + '\"'); }"
        )))
        for _ in range(25):
            if ip.user_ns.get("_CAM_IMG"):
                return _decode_dataurl(ip.user_ns.pop("_CAM_IMG"))
            time.sleep(0.2)
        return None, "browser capture unavailable here - use the image picker below"

    return None, "no webcam bridge available"


def capture_webcam():
    \"\"\"Try local camera, then browser camera. Returns (frame, error).\"\"\"
    frame, err = capture_local_webcam()
    if frame is None:
        print(f"  local camera: {err}")
        frame, err = capture_browser_webcam()
        print(f"  browser camera: {err}" if frame is None else "  browser camera: OK")
    else:
        print("  local camera: OK")
    return frame, err
"""),

    # 4 ---- add images: webcam + image picker markdown
    ("md", """## 1) Add photos: webcam + image picker

Run the next cell. It **requests the webcam** (local camera first, then the
browser's). If no camera is available, use the **image picker** button that is
shown instead — pick photos taken on your phone and they are copied into
`dataset/images/`.
"""),

    # 5 ---- capture + picker + import
    ("code", """print("Requesting webcam...")
frame, err = capture_webcam()
saved = 0
if frame is not None:
    name = os.path.join(IMG_DIR, "webcam_" + time.strftime("%Y%m%d_%H%M%S") + ".jpg")
    cv2.imwrite(name, frame)
    saved += 1
    print("Saved webcam capture:", os.path.basename(name))
    plt.figure(figsize=(6, 4))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
else:
    print(f"No webcam available ({err}). Use the image picker below.")

# Image picker: select photos to add to the dataset
import ipywidgets as widgets

uploader = widgets.FileUpload(
    accept="image/*", multiple=True, description="Pick images",
    layout=widgets.Layout(width="50%"),
)


def _on_upload(change):
    if not change.get("new"):
        return
    n = 0
    for fname, info in change["new"].items():
        data = info["content"]
        ext = os.path.splitext(fname)[1] or ".jpg"
        dst = os.path.join(IMG_DIR, "picked_" + time.strftime("%Y%m%d_%H%M%S") + "_%d" % n + ext)
        with open(dst, "wb") as f:
            f.write(data)
        n += 1
    print(f"Imported {n} image(s) into {IMG_DIR}")


uploader.observe(_on_upload, names="value")
display(uploader)
print("Pick photos now (or skip) ->")
"""),

    # 6 ---- dataset summary
    ("code", """images = list_images()
shots = sorted(glob.glob(os.path.join("dataset", "plain_screenshots", "*.jpg")))
print(f"Labeled set : {len(images)} image(s) in {IMG_DIR}")
print(f"Screenshots : {len(shots)} file(s) in dataset/plain_screenshots (unlabeled)")
if not images:
    print("NOTE: training needs LABELED frames - in the Streamlit app extract "
          "with the 'Plain screenshots' checkbox OFF.")
display(pd.DataFrame({"file": [os.path.basename(p) for p in images]}))
"""),

    # 7 ---- verify markdown
    ("md", """## 2) Verify pretrained YOLO on your classroom angles

Run each pretrained weight over every photo in `dataset/images/`, save annotated
outputs to `dataset/outputs/<model>/`, and compare how many people each model
finds (this is the ground-truth check for your camera angles).
"""),

    # 8 ---- run all models
    ("code", """def run_all_models(images, models, conf=CONF):
    \"\"\"Run each pretrained model on every image; return per-model stats.\"\"\"
    report, confs = {}, {}
    for name, weight in models.items():
        model = YOLO(weight)
        counts, conf_list = [], []
        model_out = os.path.join(OUT_DIR, name)
        os.makedirs(model_out, exist_ok=True)
        for img in images:
            results = model(img, conf=conf, verbose=False)[0]
            n = 0
            for box in results.boxes:
                if int(box.cls) != PERSON_ID:
                    continue
                n += 1
                conf_list.append(float(box.conf))
            counts.append(n)
            cv2.imwrite(os.path.join(model_out, os.path.basename(img)),
                        results.plot())  # boxes + pose landmarks
        report[name] = counts
        confs[name] = conf_list
        if counts:
            print(f"  {name:14s} avg={np.mean(counts):.1f}  min={np.min(counts)} max={np.max(counts)}")
        else:
            print(f"  {name:14s} no images processed")
    return report, confs

report, confs = {}, {}
vis_images, vis_src = list_visual_images()
if not vis_images:
    print("No images found anywhere. Open the Streamlit app -> upload a video -> "
          "'Break video into training frames' (Plain OFF = training data, "
          "ON = screenshots), then rerun this cell.")
else:
    images = vis_images          # downstream visual cells use the same set
    print(f"Visual check on {len(vis_images)} image(s) - source: {vis_src}")
    report, confs = run_all_models(images, PRETRAINED)
"""),

    # 9 ---- comparison table + charts
    ("code", """rows = []
for name, counts in report.items():
    rows.append({
        "model": name,
        "avg_persons": round(float(np.mean(counts)), 2),
        "min": int(np.min(counts)),
        "max": int(np.max(counts)),
        "detections": int(len(confs[name])),
        "mean_conf": round(float(np.mean(confs[name])), 3) if confs[name] else 0.0,
    })
rows.sort(key=lambda r: -r["avg_persons"])
display(pd.DataFrame(rows))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
ax1.bar([r["model"] for r in rows], [r["avg_persons"] for r in rows], color="steelblue")
ax1.set_ylabel("avg persons detected / image")
ax1.set_title("Pretrained YOLO on classroom photos (various angles)")
for name, cl in confs.items():
    if cl:
        ax2.hist(cl, bins=20, alpha=0.5, label=name)
ax2.set_xlabel("confidence")
ax2.set_ylabel("count")
ax2.set_title("Person detection confidence")
ax2.legend()
plt.tight_layout()
plt.show()
"""),

    # 10 ---- side by side angle check
    ("code", """def show_side_by_side(names, num=3):
    \"\"\"Original + each model output, to visually verify classroom angles.\"\"\"
    for img in images[:num]:
        fig, axes = plt.subplots(1, len(names) + 1, figsize=(4 * (len(names) + 1), 4))
        axes[0].imshow(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB))
        axes[0].set_title("original")
        axes[0].axis("off")
        for ax, name in zip(axes[1:], names):
            out = os.path.join(OUT_DIR, name, os.path.basename(img))
            if os.path.exists(out):
                ax.imshow(cv2.cvtColor(cv2.imread(out), cv2.COLOR_BGR2RGB))
            ax.set_title(name)
            ax.axis("off")
        plt.suptitle(os.path.basename(img) + "  (classroom angle check)")
        plt.tight_layout()
        plt.show()

show_side_by_side(["yolo11n", "yolov8n-pose", "yolo11n-pose"])
"""),

    # 11 ---- pose visualization
    ("md", """### Pose landmarks

The pose model gives 17 body keypoints per person in addition to the count —
useful later for *occupied vs empty* seat analysis (e.g. keypoint height).
"""),

    # 12 ---- pose viz code
    ("code", """pose = YOLO("models/pretrained/yolo11n-pose.pt")
if images:
    res = pose(images[0], conf=CONF, verbose=False)[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    axes[0].imshow(cv2.cvtColor(cv2.imread(images[0]), cv2.COLOR_BGR2RGB))
    axes[0].set_title("original classroom photo")
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB))
    axes[1].set_title("YOLO11-pose: person boxes + landmarks")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()
    n = len([b for b in res.boxes if int(b.cls) == PERSON_ID])
    print(f"People detected in sample: {n}")
"""),

    # 13 ---- chair detection markdown
    ("md", """## 2b) Find the chairs + chair-aware occupancy

`detect_live.py` / `app.py` count people ONLY if they are inside the **chair
area**. So we first have to know the chairs:

1. YOLO detects every chair (COCO class 56) and the **total seats** is the number
   of chairs found.
2. A **static box** is drawn around the whole chair area.
3. A person counts as occupied only when their box center is inside that box.

The cells below verify this on your photos.
"""),

    # 14 ---- chair detection
    ("code", """def run_chair_detection(images, chair_model="models/pretrained/yolo11n.pt", conf=0.25):
    \"\"\"Count chairs + chair-area ROI for every image.\"\"\"
    model = YOLO(chair_model)
    chair_out = os.path.join(OUT_DIR, "chairs")
    os.makedirs(chair_out, exist_ok=True)
    counts, rois = [], []
    for img in images:
        res = model(img, conf=conf, classes=[CHAIR_ID], verbose=False)[0]
        union, n = None, 0
        if res.boxes is not None and len(res.boxes) > 0:
            arr = np.array([b.xyxy[0].cpu().numpy() for b in res.boxes])
            n = len(arr)
            union = (arr[:, 0].min(), arr[:, 1].min(), arr[:, 2].max(), arr[:, 3].max())
        counts.append(n)
        rois.append(union)
        vis = res.plot()
        if union is not None:
            x1, y1, x2, y2 = map(int, union)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(vis, f"CHAIR AREA ({n})", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(os.path.join(chair_out, os.path.basename(img)), vis)
    return counts, rois

chair_counts, chair_rois = run_chair_detection(images)

chair_df = pd.DataFrame({
    "file": [os.path.basename(p) for p in images],
    "chairs_found": chair_counts,
})
display(chair_df)

found = [c for c in chair_counts if c > 0]
seat_total = int(np.median(found)) if found else 0
print("Recommended SEAT TOTAL (median chairs detected) =", seat_total)
print("Annotated chair outputs saved to dataset/outputs/chairs/")
"""),

    # 15 ---- chair-aware occupancy check
    ("code", """def chair_aware_occupancy(images, chair_rois, person_model="models/pretrained/yolo11n.pt",
                            conf=0.30, show=3):
    \"\"\"Count people whose center is inside the chair-area box.\"\"\"
    model = YOLO(person_model)
    summary = []
    show = min(show, len(images))
    fig, axes = plt.subplots(show, 2, figsize=(10, 3 * show),
                             squeeze=False) if show else (None, None)
    for i, img in enumerate(images):
        res = model(img, conf=conf, verbose=False)[0]
        occ = 0
        roi = chair_rois[i]
        if roi is not None:
            x1r, y1r, x2r, y2r = roi
            for b in res.boxes:
                if int(b.cls) != PERSON_ID:
                    continue
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                if x1r <= (x1 + x2) / 2 <= x2r and y1r <= (y1 + y2) / 2 <= y2r:
                    occ += 1
        summary.append({"file": os.path.basename(img), "occupied_in_chair_area": occ})
        if i < show and roi is not None:
            vis = res.plot()
            x1r, y1r, x2r, y2r = map(int, roi)
            cv2.rectangle(vis, (x1r, y1r), (x2r, y2r), (0, 255, 255), 2)
            axes[i, 0].imshow(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB))
            axes[i, 0].set_title("original")
            axes[i, 0].axis("off")
            axes[i, 1].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            axes[i, 1].set_title(f"occupied in chair area = {occ}")
            axes[i, 1].axis("off")
    if show:
        plt.tight_layout()
        plt.show()
    return summary

if chair_rois and any(r is not None for r in chair_rois):
    occ_df = pd.DataFrame(chair_aware_occupancy(images, chair_rois))
    display(occ_df)
else:
    print("No chairs detected in the dataset - check the photos / camera angle.")
"""),

    # 16 ---- train section markdown
    ("md", """## 3) Train **class-monitor-ai** (transfer learning on your dataset)

This is the **image loop**: the model learns from every photo in the dataset.

1. **Auto-label** — any image without a label file gets labelled automatically by
   a pretrained model (person boxes -> `dataset/labels/*.txt`).
2. **Split** — a fraction of images move to `dataset/val/` for validation.
3. **Fine-tune** — transfer learning: `yolo11n.pt` (COCO-pretrained) is
   fine-tuned on *your* photos for a **low number of epochs** (10) so you get a
   nice result quickly.
4. **Store** — the trained model is saved as `models/tuned/class-monitor-ai.pt`
   (+ a `.meta.json` with the metrics). The Streamlit app picks it up
   automatically in its model dropdown.

If you already labelled photos by hand, your labels are used as-is.
"""),

    # 14 ---- auto-label loop
    ("code", """def auto_label_images(label_model="models/pretrained/yolo11n.pt", conf=0.30):
    \"\"\"Loop over every image; if it has no label file, run YOLO and save person boxes.\"\"\"
    model = YOLO(label_model)
    need = [p for p in list_images()
            if not os.path.exists(os.path.join(LABEL_DIR,
                                               os.path.splitext(os.path.basename(p))[0] + ".txt"))]
    print(f"{len(list_images()) - len(need)} already labelled, {len(need)} to auto-label with {label_model}")
    for p in need:
        res = model(p, conf=conf, verbose=False)[0]
        ih, iw = res.orig_shape  # (height, width)
        lines = []
        for b in res.boxes:
            if int(b.cls) != PERSON_ID:
                continue
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
            cx, cy = ((x1 + x2) / 2) / iw, ((y1 + y2) / 2) / ih
            w, h = (x2 - x1) / iw, (y2 - y1) / ih
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        txt = os.path.join(LABEL_DIR, os.path.splitext(os.path.basename(p))[0] + ".txt")
        with open(txt, "w") as f:
            f.write("\\n".join(lines) + "\\n")
    print("Auto-labelling finished.")


if glob.glob(os.path.join(LABEL_DIR, "*.txt")):
    print("Labels already exist - skipping auto-label (your hand labels are used).")
else:
    auto_label_images()
"""),

    # 15 ---- train/val split + custom.yaml
    ("code", """def make_split(val_frac=0.2, seed=42, min_val=2):
    \"\"\"Move a fraction of images + labels into dataset/val for validation.\"\"\"
    for d in (VAL_IMG_DIR, VAL_LAB_DIR):
        os.makedirs(d, exist_ok=True)
        for f in glob.glob(os.path.join(d, "*")):
            os.remove(f)
    imgs = list_images()
    n_val = max(min_val, int(len(imgs) * val_frac))
    rng = np.random.default_rng(seed)
    val = set(rng.choice(imgs, size=min(n_val, len(imgs)), replace=False))
    moved = 0
    for p in imgs:
        if p not in val:
            continue
        base = os.path.splitext(os.path.basename(p))[0]
        shutil.copy(p, os.path.join(VAL_IMG_DIR, os.path.basename(p)))
        lab = os.path.join(LABEL_DIR, base + ".txt")
        if os.path.exists(lab):
            shutil.copy(lab, os.path.join(VAL_LAB_DIR, base + ".txt"))
            os.remove(lab)
        os.remove(p)
        moved += 1
    print(f"Train/val split done: {len(list_images())} train, {moved} val.")


make_split()

with open("custom.yaml", "w") as f:
    f.write("path: .\\ntrain: dataset/images\\nval: dataset/val/images\\nnames:\\n  0: person\\n")
print("Wrote custom.yaml ->", os.path.abspath("custom.yaml"))
"""),

    # 16 ---- train
    ("code", """# Transfer learning -> trains YOUR model: class-monitor-ai
EPOCHS = 40      # small datasets need many passes before class scores calibrate
IMGSZ = 640
base_model = "models/pretrained/yolo11n.pt"     # COCO-pretrained = transfer learning
# ---- Preflight: NEVER block - bootstrap labels if missing ------------------
import glob as _glob
import shutil as _sh

_stems = [os.path.splitext(os.path.basename(p))[0]
          for p in _glob.glob("dataset/images/*.jpg")]
_pairs = sum(1 for s in _stems if os.path.exists(f"dataset/labels/{s}.txt"))

if _pairs == 0:
    _shots = sorted(_glob.glob("dataset/plain_screenshots/*.jpg"))
    _imgs = sorted(_glob.glob("dataset/images/*.jpg"))
    if not _imgs and not _shots:
        raise SystemExit(
            "TRAINING STOPPED: both dataset/images and "
            "dataset/plain_screenshots are EMPTY - there is nothing to "
            "train on. Extract frames first (Streamlit app -> upload video "
            "-> Process video), then rerun this cell.")
    os.makedirs("dataset/images", exist_ok=True)
    os.makedirs("dataset/labels", exist_ok=True)
    _moved = 0
    for _p in _shots:              # promote screenshots into the train set
        _dst = os.path.join("dataset", "images", os.path.basename(_p))
        if not os.path.exists(_dst):
            _sh.copy2(_p, _dst)
            _moved += 1
    print(f"No labels found -> AUTO-LABELING with {base_model} (person) "
          f"so training can run anyway... (promoted {_moved} screenshot(s))")
    _boot = YOLO(base_model)
    _made = 0
    for _p in sorted(_glob.glob("dataset/images/*.jpg")):
        _stem = os.path.splitext(os.path.basename(_p))[0]
        _lp = f"dataset/labels/{_stem}.txt"
        if os.path.exists(_lp):
            continue
        _img = cv2.imread(_p)
        _h, _w = _img.shape[:2]
        _r = _boot(_p, conf=0.30, classes=[PERSON_ID], verbose=False)[0]
        _lines = []
        for _b in _r.boxes:
            _x1, _y1, _x2, _y2 = _b.xyxy[0].tolist()
            _lines.append(f"0 {(_x1 + _x2) / 2 / _w:.6f} "
                          f"{(_y1 + _y2) / 2 / _h:.6f} "
                          f"{(_x2 - _x1) / _w:.6f} {(_y2 - _y1) / _h:.6f}")
        with open(_lp, "w") as _f:
            _f.write("\\n".join(_lines) + ("\\n" if _lines else ""))
        _made += 1
    print(f"Auto-labeled {_made} image(s). NOTE: these are pretrained-model "
          "person guesses - hand-correct later for higher accuracy.")
    _pairs = _made

# ---- ensure the VALIDATION split exists (self-healing) ----------------------
_val_imgs = sorted(_glob.glob("dataset/val/images/*.jpg"))
if not _val_imgs:
    os.makedirs("dataset/val/images", exist_ok=True)
    os.makedirs("dataset/val/labels", exist_ok=True)
    _all = sorted(_glob.glob("dataset/images/*.jpg"))
    _n_val = max(1, len(_all) // 5)          # ~20% of train, at least 1
    for _p in _all[-_n_val:]:                # tail = most recent frames
        _stem = os.path.splitext(os.path.basename(_p))[0]
        _sh.move(_p, f"dataset/val/images/{os.path.basename(_p)}")
        _lp = f"dataset/labels/{_stem}.txt"
        if os.path.exists(_lp):
            _sh.move(_lp, f"dataset/val/labels/{_stem}.txt")
    print(f"Validation set was empty -> moved {_n_val} image(s) into "
          "dataset/val/ (~20% split)")

for _cache in ("dataset/labels.cache", "dataset/val/labels.cache"):
    if os.path.exists(_cache):
        os.remove(_cache)   # stale cache hash crashed the old run
print(f"Preflight OK: {len(_glob.glob('dataset/images/*.jpg'))} train / "
      f"{len(_glob.glob('dataset/val/images/*.jpg'))} val labeled image(s)")

# ---- guard: kernel must hold the SAME ultralytics that is on disk -----------
import ultralytics as _ul
import importlib.metadata as _im

_disk_ver = _im.version("ultralytics")
if _disk_ver != _ul.__version__:
    raise SystemExit(
        f"STOPPED: ultralytics {_ul.__version__} is loaded in this notebook "
        f"but version {_disk_ver} is installed on disk (it was downgraded to "
        "fix a training bug). Mixed old+new modules crash training with "
        "'RandomPerspective has no attribute pre_transform'. "
        "FIX: menu Kernel -> Restart Kernel, then rerun the cells above "
        "and this one.")

model = YOLO(base_model)
results = model.train(
    data="custom.yaml",
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=4,
    freeze=10,          # KEEP the COCO backbone frozen -> transfer learning
                        # actually transfers (full unfreeze on a tiny set made
                        # the last run forget everything and detect nothing)
    optimizer="SGD",    # explicit so lr0 below is honored ('auto' overrides it)
    lr0=0.01,           # head-only training needs a normal LR to lift its class
                        # scores - gentle 0.001 left them stuck near 0.03 forever
    mosaic=0.0,         # no 4-image mixing - too few images for it to help
    workers=0,          # Windows+Jupyter: spawned workers crash with
                        # 'DataLoader worker exited unexpectedly' - load
                        # frames in-process instead (tiny set = no slowdown)
    patience=max(3, EPOCHS // 2),
    project="models/training",
    name="class-monitor-ai",
    device=device,
    verbose=True,
)

# Store the trained model where the Streamlit app expects it
import json, shutil

best = os.path.join(results.save_dir, "weights", "best.pt")
os.makedirs("models/tuned", exist_ok=True)
shutil.copy2(best, "models/tuned/class-monitor-ai.pt")

# ---- auto-calibrate the app's confidence threshold on the val split ---------
# tiny fine-tunes rank people almost perfectly (high mAP50) but their class
# scores stay LOW - so the app must use a lower conf than the usual 0.25.
_tuned = YOLO(best)
_val_pairs = {}
for _p in sorted(_glob.glob("dataset/val/images/*.jpg")):
    _s = os.path.splitext(os.path.basename(_p))[0]
    _lp = f"dataset/val/labels/{_s}.txt"
    if os.path.exists(_lp):
        _val_pairs[_p] = sum(1 for _ln in open(_lp) if _ln.strip())
conf_floor, best_f1 = 0.25, 0.0
_gt_total = sum(_val_pairs.values()) or 1
for _c in [round(0.01 * i, 2) for i in range(1, 41)]:
    _tp = _fp = 0
    for _p, _n in _val_pairs.items():
        _pred = len(_tuned(_p, conf=_c, classes=[PERSON_ID],
                           verbose=False)[0].boxes)
        _tp += min(_pred, _n)
        _fp += max(_pred - _n, 0)
    _prec = _tp / (_tp + _fp) if (_tp + _fp) else 0.0
    _rec = _tp / _gt_total
    _f1 = 2 * _prec * _rec / (_prec + _rec) if (_prec + _rec) else 0.0
    if _f1 > best_f1 and _rec >= 0.5:   # never sacrifice recall for precision
        best_f1, conf_floor = _f1, _c
print(f"Auto-calibrated app confidence threshold: {conf_floor} "
      f"(F1={best_f1:.2f} on {len(_val_pairs)} val image(s))")

meta = {
    "name": "class-monitor-ai",
    "base_weights": base_model,
    "data": "custom.yaml",
    "classes": model.names,
    "epochs": EPOCHS,
    "imgsz": IMGSZ,
    "metrics": {k: float(v) for k, v in results.results_dict.items()},
    "run_dir": str(results.save_dir),
    "conf_floor": float(conf_floor),
}
with open("models/tuned/class-monitor-ai.meta.json", "w") as f:
    json.dump(meta, f, indent=2, default=str)

print("Saved : models/tuned/class-monitor-ai.pt")
_map50 = float(meta["metrics"].get("metrics/mAP50(B)", 0.0) or 0.0)
print(f"mAP50 : {_map50:.3f}")
if _map50 < 0.3:
    print()
    print("WARNING: this model is WEAK and will miss people.")
    print("It was trained on a small auto-labeled set. To improve it:")
    print("  1. Streamlit app -> upload video(s) from different angles")
    print("  2. 'Break video into training frames' ON, 'Plain screenshots' OFF")
    print("  3. Rerun this training cell (EPOCHS=40)")
    print("Until then keep the app on a PRETRAINED model.")
else:
    print("The Streamlit app will use the auto-calibrated threshold above "
          "automatically when this model is selected.")
print("app.py now lists it in the model dropdown as 'class-monitor-ai (my trained model)'.")
"""),

    # 17 ---- evaluate tuned model (tuned vs pretrained, side by side)
    ("code", """import json
tuned = YOLO("models/tuned/class-monitor-ai.pt")
_base = YOLO(base_model)
try:
    _meta = json.load(open("models/tuned/class-monitor-ai.meta.json"))
except Exception:
    _meta = {}
_floor = float(_meta.get("conf_floor") or 0.25)
print(f"Evaluating class-monitor-ai at its CALIBRATED threshold "
      f"{_floor} (tiny fine-tunes score low but rank people well)")
vis_images, vis_src = list_visual_images()
if not vis_images:
    print("No frames to preview - extract some in the Streamlit app first.")
else:
    _show = vis_images[:3]
    print(f"Preview on {len(_show)} frame(s) - source: {vis_src}")
    _fig, _axes = plt.subplots(len(_show), 2, figsize=(11, 5 * len(_show)),
                               squeeze=False)
    for _row, _p in zip(_axes, _show):
        _rb = _base(_p, conf=0.25, classes=[PERSON_ID], verbose=False)[0]
        _rt = tuned(_p, conf=_floor, classes=[PERSON_ID], verbose=False)[0]
        _nb, _nt = len(_rb.boxes), len(_rt.boxes)
        _row[0].imshow(cv2.cvtColor(_rb.plot(), cv2.COLOR_BGR2RGB))
        _row[0].set_title(f"pretrained yolo11n: {_nb} people")
        _row[0].axis("off")
        _row[1].imshow(cv2.cvtColor(_rt.plot(), cv2.COLOR_BGR2RGB))
        _row[1].set_title(f"class-monitor-ai: {_nt} people @{_floor}")
        _row[1].axis("off")
    plt.tight_layout()
    plt.show()
    _tuned_counts = [len(tuned(v, conf=_floor, classes=[PERSON_ID],
                               verbose=False)[0].boxes) for v in vis_images]
    if sum(_tuned_counts) == 0:
        print()
        print("class-monitor-ai found 0 people even at its calibrated "
              "threshold.")
    elif any(c > 60 for c in _tuned_counts):
        print()
        print(f"class-monitor-ai is OVER-firing (up to {max(_tuned_counts)} "
              "boxes/frame vs ~10-15 real people): the transfer worked - it ")
        print("detects person-shapes - but there are not enough labeled "
              "frames yet to calibrate its scores. To sharpen it:")
    elif max(_tuned_counts) < 3:
        print()
        print("class-monitor-ai barely fires - it needs more labeled frames:")
    else:
        print()
        print(f"class-monitor-ai looks USABLE ({min(_tuned_counts)}-"
              f"{max(_tuned_counts)} boxes/frame). Select it in the Streamlit "
              "app - the confidence is applied automatically.")
    if sum(_tuned_counts) == 0 or any(c > 60 for c in _tuned_counts) \\
            or max(_tuned_counts) < 3:
        print("  1. Streamlit app -> upload video(s) from DIFFERENT angles")
        print("  2. 'Break video into training frames' ON, 'Plain screenshots' OFF")
        print("     -> aim for 30+ labeled frames covering the whole room")
        print("  3. Rerun the training cell")
        print("Until then keep the app on a PRETRAINED model (it already "
              "knows 'person' from 118k COCO images).")
"""),

    # 18 ---- wrap up
    ("md", """## Wrap up

- **Verify** tells you which pretrained weight transfers best to your classroom
  angles (look at `avg_persons` + the annotated photos in `dataset/outputs/`).
- **Train** produced your own model: `models/tuned/class-monitor-ai.pt`
  (transfer-learned from `yolo11n.pt` on *your* dataset).
- Use it in the Streamlit app - once the file exists, the model dropdown shows
  **"class-monitor-ai (trained)"** and it is used for person/head tracking.
- The app's **"Count these as seats" prompt** (e.g. `chair, monitor`) decides
  what gets boxed and counted as a seat.
"""),
]


def to_cell(cell_type, source):
    lines = source.split("\n")
    if cell_type == "code" and lines and lines[-1] != "":
        lines.append("")
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines[-1] else []),
        "execution_count": None if cell_type == "code" else None,
        "outputs": [] if cell_type == "code" else [],
    }


nb = {
    "cells": [to_cell(*c) for c in CELLS],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "verify_yolo.ipynb"))
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Wrote", os.path.abspath(out_path))
