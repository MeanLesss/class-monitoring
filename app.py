"""
Classroom Occupancy - Streamlit web app (live browser webcam).

Detection logic:
- A POSE model finds each person and, from the face keypoints (nose/eyes/ears),
  builds a HEAD box and an UPPER-BODY box (no full-body box).
- Chairs (COCO class 56) are detected with a light detect model every ~2 s (they
  don't move), so it costs almost nothing.
- A person counts as OCCUPIED only if their HEAD center is inside the chair area.

Run (from this folder):
    .\\ai-venv\\Scripts\\streamlit run app.py --server.address 0.0.0.0 --server.port 8501
Then on your phone (same Wi-Fi) open:  http://<PC_IP>:8501
"""

import asyncio
import csv
import gc
import json
import os
import re
import tempfile
import threading
import time
import traceback

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from seatdraw import seatdraw

from ultralytics import YOLO

# --------------------------------------------------------------------------
# streamlit-webrtc compat patch: on some Streamlit versions the component's
# internal on_change callback reads st.session_state[frontend_key] before the
# frontend has written it, raising a KeyError on START. Wrap it so that race
# is ignored (the value arrives on the normal component return path).
# --------------------------------------------------------------------------
import streamlit_webrtc.component as _swc_component

_swc_orig_cb = _swc_component._make_state_change_callback


def _swc_safe_callback(key, frontend_key, user_on_change):
    cb = _swc_orig_cb(key, frontend_key, user_on_change)

    def safe():
        try:
            cb()
        except KeyError:
            pass  # frontend value not ready yet - normal path will handle it

    return safe


_swc_component._make_state_change_callback = _swc_safe_callback

if os.name == "nt":
    # Fix benign Windows Proactor teardown error on WebRTC disconnect:
    # "Exception in callback _ProactorBasePipeTransport._call_connection_lost()"
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

PERSON_ID = 0   # COCO class id for "person"
CHAIR_ID = 56   # COCO class id for "chair"
DESK_ID = 60    # COCO class id for "dining table" (used as the desk proxy)
TV_ID = 62      # COCO class id for "tv" (shown as MONITOR - no 'monitor' class exists)
FURN_CONF = 0.22    # low confidence floor for the furniture scan (small/distant items)
FURN_IMGSZ = 640    # fixed inference size for the furniture scan (finds small screens)

CHAIR_MODEL = "models/pretrained/yolo11n.pt"
TRAINED_MODEL = "models/tuned/class-monitor-ai.pt"   # trained in verify_yolo.ipynb

# Prompt words that are not exact COCO names get mapped to these
PROMPT_SYNONYMS = {
    "monitor": ["tv"], "screens": ["tv"], "screen": ["tv"],
    "computers": ["laptop", "tv"], "computer": ["laptop"], "pc": ["laptop"],
    "desk": ["dining table"], "desks": ["dining table"],
    "table": ["dining table"], "tables": ["dining table"],
    "sofa": ["couch"], "sofas": ["couch"],
    "phone": ["cell phone"], "phones": ["cell phone"],
}

# Latest processed live frame + manual seat boxes drawn on it (module globals,
# shared between the WebRTC worker thread and the main script). Created ONCE -
# Streamlit reruns the script, so re-creating them would wipe the camera thread's
# latest frame and the drawn boxes.
if "_LIVE" not in globals():
    _LIVE = {"frame": None}
if "live_manual_rois" not in globals():
    live_manual_rois = []
if "live_box_wh" not in globals():
    live_box_wh = None
if "live_prompt_hits" not in globals():
    live_prompt_hits = []
if "_LAST_DET" not in globals():
    _LAST_DET = {"res": None}      # last person-detection results (frame extraction)

# Single source of truth for the seat boxes is st.session_state (reliable across
# reruns). The values are mirrored into module globals that the WebRTC worker
# thread can read.
live_manual_rois = st.session_state.get("seat_boxes", [])
live_box_wh = st.session_state.get("seat_boxes_wh")

# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
st.set_page_config(page_title="Classroom Occupancy", page_icon="🎓")
st.title("Classroom Occupancy")
st.caption("Upper-body + head (facial) tracking; counts heads inside the seat area.")

with st.sidebar:
    st.header("Settings")
    _model_labels = {
        "models/pretrained/yolo11n-pose.pt": "yolo11n-pose (pretrained)",
        "models/pretrained/yolov8n-pose.pt": "yolov8n-pose (pretrained)",
        "models/pretrained/yolo11n.pt": "yolo11n (pretrained)",
        "models/pretrained/yolov8n.pt": "yolov8n (pretrained)",
    }
    if os.path.exists(TRAINED_MODEL):
        _model_labels[TRAINED_MODEL] = "class-monitor-ai (my trained model)"
    model_name = st.selectbox(
        "Person model (pose = upper body + head)",
        list(_model_labels),
        format_func=lambda p: _model_labels.get(p, p),
        help="Pose models give upper-body + head/facial tracking. "
             "'class-monitor-ai' is trained in verify_yolo.ipynb.",
    )
    if os.path.exists(TRAINED_MODEL):
        try:
            _meta = json.load(open("models/tuned/class-monitor-ai.meta.json"))
            _map50 = float(_meta.get("metrics", {})
                           .get("metrics/mAP50(B)", 0.0) or 0.0)
        except Exception:
            _map50 = None
        try:
            _conf_floor = float(_meta.get("conf_floor") or 0) or None
        except Exception:
            _conf_floor = None
        if model_name == TRAINED_MODEL and (_map50 is None or _map50 < 0.3):
            st.warning(
                f"class-monitor-ai is weak (mAP50 "
                f"{_map50 if _map50 is not None else 'n/a'}) - it will miss "
                "people. It was trained on very few auto-labeled frames. "
                "Stay on a PRETRAINED model, extract more labeled frames "
                "(video -> 'Break video into training frames', Plain OFF), "
                "then retrain in verify_yolo.ipynb.")
        elif model_name == TRAINED_MODEL:
            st.caption(f"Your trained model - mAP50 {_map50:.2f}")
    else:
        _conf_floor = None
    conf = st.slider("Confidence", 0.1, 0.9, 0.4, 0.05)
    if (model_name == TRAINED_MODEL and _conf_floor
            and _map50 is not None and _map50 >= 0.3 and _conf_floor < conf):
        conf = _conf_floor
        st.caption(f"confidence auto-set to {conf} for your tuned model "
                   "(small fine-tunes score low but rank people well)")
    seats = st.number_input("Total seats (0 = auto from seats found)", 0, 200, 0)
    imgsz = st.select_slider("Speed (inference size)", [320, 416, 480, 640], 480,
                             help="Lower = faster FPS, slightly less detail.")
    st.divider()
    seat_prompt = st.text_input(
        "Count these as seats",
        value="",
        placeholder="e.g. chair, monitor, laptop",
        help="Comma-separated object types to detect, box and count as SEATS "
             "(each detection = 1 seat). Heads inside those boxes count as "
             "occupied. Leave empty for the automatic chair/desk area.",
    )

    st.divider()
    st.write("**How it works**")
    if seat_prompt.strip():
        st.write(f"1. Detects every **{seat_prompt}** -> each box is one seat.")
    else:
        st.write("1. Finds the chairs (every ~2 s) and locks the seat area box.")
    st.write("2. Tracks each person's HEAD (nose/eyes/ears) + upper body.")
    st.write("3. Only heads inside a seat box count as occupied.")


# --------------------------------------------------------------------------
# Models - loaded EAGERLY in the main thread (never inside the WebRTC worker
# thread, which has no Streamlit context and can deadlock on st.cache_resource).
# --------------------------------------------------------------------------
_MODEL_CACHE = {}
_MODEL_LOCK = threading.Lock()


def get_model(path):
    model = _MODEL_CACHE.get(path)
    if model is None:
        with _MODEL_LOCK:
            model = _MODEL_CACHE.get(path)
            if model is None:
                model = YOLO(path)
                _MODEL_CACHE[path] = model
    return model


get_model(model_name)          # person (pose) model
get_model(CHAIR_MODEL)         # chair model

IS_POSE = "pose" in model_name.lower()


def _warmup(model):
    """Run one tiny inference in the MAIN thread so the WebRTC worker thread
    never has to initialize CUDA / JIT (that used to stall the stream dead)."""
    try:
        model.predict(np.zeros((240, 320, 3), dtype=np.uint8),
                      conf=0.9, imgsz=320, verbose=False)
    except Exception:
        pass


# Warm up ONCE (not on every Streamlit rerun - each rerun used to waste a GPU
# inference and contributed to the video pausing while drawing).
if "_WARMED" not in globals():
    _WARMED = True
    _warmup(get_model(model_name))
    _warmup(get_model(CHAIR_MODEL))

# Serialize YOLO calls: streamlit-webrtc may invoke recv() from several threads,
# and concurrent CUDA inference on the same model object can hang/crash.
_INFER_LOCK = threading.Lock()


# --------------------------------------------------------------------------
# Seat-object prompt ("chair, monitor, laptop") -> COCO class ids
# --------------------------------------------------------------------------
_PROMPT_CACHE = {}


def resolve_seat_prompt(text):
    """Comma-separated words -> ([(class_id, label)...], [unmatched words]).
    Matched against the COCO names of the pretrained chair model using exact,
    singular/plural, synonym and substring matching."""
    if not text or not text.strip():
        return [], []
    key = " ".join(text.lower().split())
    if key in _PROMPT_CACHE:
        return _PROMPT_CACHE[key]
    id_by_name = {str(v).lower(): int(k)
                  for k, v in get_model(CHAIR_MODEL).names.items()}
    hits, missing, seen = [], [], set()
    for raw in re.split(r"[,;\n]+", text):
        w = raw.strip().lower()
        if not w:
            continue
        w2 = w[:-1] if w.endswith("s") and not w.endswith("ss") else w  # chairs->chair
        cands = [w, w2] + [s for k in (w, w2) for s in PROMPT_SYNONYMS.get(k, [])]
        hit = None
        for c in cands:
            if c and c in id_by_name:
                hit = (id_by_name[c], c)
                break
        if hit is None:                       # substring fallback (both ways)
            for n, i in id_by_name.items():
                if w in n or n in w:
                    hit = (i, n)
                    break
        if hit is None:
            missing.append(raw.strip())
        elif hit[0] not in seen:
            seen.add(hit[0])
            hits.append(hit)
    _PROMPT_CACHE[key] = (hits, missing)
    return hits, missing


seat_hits, seat_missing = resolve_seat_prompt(seat_prompt)
live_prompt_hits = seat_hits          # read by the WebRTC worker thread

if seat_missing:
    st.warning("Ignored (no such object class): "
               + ", ".join(seat_missing)
               + ". Try e.g. chair, tv, laptop, dining table, couch, keyboard.")
elif seat_hits:
    st.caption("Seat objects: **" + ", ".join(lbl for _, lbl in seat_hits) + "**")


def head_and_upper(kpts, conf_thr):
    """From 17 pose keypoints build (head_box, upper_body_box, face_points)."""
    pts = {i: (float(kpts[i][0]), float(kpts[i][1]))
           for i in range(17) if float(kpts[i][2]) > conf_thr}
    if 0 not in pts:  # no nose keypoint -> face not visible
        return None, None, []

    face = [pts[i] for i in (0, 1, 2, 3, 4) if i in pts]      # nose, eyes, ears
    xs = [p[0] for p in face]
    ys = [p[1] for p in face]
    w = max(max(xs) - min(xs), 30.0)
    h = max(max(ys) - min(ys), 30.0)
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    head = (int(cx - 0.7 * w), int(cy - 0.7 * h),
            int(cx + 0.7 * w), int(cy + 0.8 * h))

    body = [pts[i] for i in (0, 1, 2, 3, 4, 5, 6, 11, 12) if i in pts]
    upper = (int(min(p[0] for p in body)), int(min(p[1] for p in body)),
             int(max(p[0] for p in body)), int(max(p[1] for p in body)))
    return head, upper, [(int(p[0]), int(p[1])) for p in face]


# --------------------------------------------------------------------------
# Shared drawing helpers (used by the live stream AND the image upload)
# --------------------------------------------------------------------------
def _draw_people(img, results, rois, conf_thr):
    """Draw upper-body + head boxes; returns occupied count (heads in ANY seat box)."""
    keyp = results.keypoints
    occupied = 0
    for i, b in enumerate(results.boxes):
        if int(b.cls) != PERSON_ID:
            continue
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)

        head, upper, face = (None, None, [])
        if IS_POSE and keyp is not None and len(keyp) > i:
            head, upper, face = head_and_upper(keyp.data[i].cpu().numpy(), conf_thr)
        if upper is None:
            upper = (x1, y1, x2, int(y1 + 0.6 * (y2 - y1)))
        if head is None:
            head = (x1, y1, x2, int(y1 + 0.3 * (y2 - y1)))

        hcx, hcy = (head[0] + head[2]) / 2, (head[1] + head[3]) / 2
        inside = any(x1r <= hcx <= x2r and y1r <= hcy <= y2r
                     for (x1r, y1r, x2r, y2r) in rois)
        color = (0, 255, 0) if inside else (140, 140, 140)

        cv2.rectangle(img, (upper[0], upper[1]), (upper[2], upper[3]), color, 2)
        cv2.putText(img, f"p{float(b.conf):.2f}", (upper[0], upper[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.rectangle(img, (head[0], head[1]), (head[2], head[3]), (255, 255, 0), 2)
        for fx, fy in face:
            cv2.circle(img, (fx, fy), 2, (0, 255, 255), -1)
        if face:
            cv2.putText(img, "FACE", (head[0], head[3] + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        if inside:
            occupied += 1
    return occupied


def _draw_furniture(img, chair_boxes, desk_boxes, tv_boxes):
    """Color-coded box + NAME #n tag on every detected furniture item."""
    def _tagged(boxes, color, label):
        if boxes is None or len(boxes) == 0:
            return
        for n, (x1, y1, x2, y2) in enumerate(
                np.asarray(boxes, dtype=int).reshape(-1, 4), 1):
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            cv2.putText(img, f"{label} #{n}", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
    _tagged(chair_boxes, (255, 170, 0), "CHAIR")
    _tagged(desk_boxes, (255, 0, 255), "DESK")
    _tagged(tv_boxes, (0, 200, 255), "MONITOR")


def _furniture_area(img):
    """Detect chairs + desks + monitors (tv). Returns
    (chair_boxes, desk_boxes, tv_boxes, roi_or_None, seat_count, items).
    roi is None when NO furniture was found - no whole-frame fallback.
    items: [(name, box, conf)] audit list used for the CSV export."""
    res = get_model(CHAIR_MODEL)(img, conf=FURN_CONF,
                                 classes=[CHAIR_ID, DESK_ID, TV_ID],
                                 imgsz=FURN_IMGSZ, verbose=False)[0]
    chairs, desks, tvs, items = [], [], [], []
    for b in res.boxes:
        box = b.xyxy[0].cpu().numpy()
        cf = float(b.conf)
        cid = int(b.cls)
        if cid == CHAIR_ID:
            chairs.append(box)
            items.append(("chair", box, cf))
        elif cid == DESK_ID:
            desks.append(box)
            items.append(("desk", box, cf))
        elif cid == TV_ID:
            tvs.append(box)
            items.append(("monitor", box, cf))
    all_boxes = chairs + desks + tvs
    roi = None
    if all_boxes:
        arr = np.array(all_boxes)
        x1, y1 = max(0, int(arr[:, 0].min())), max(0, int(arr[:, 1].min()))
        x2, y2 = int(arr[:, 2].max()), int(arr[:, 3].max())
        m = max(8, int(0.02 * (x2 - x1)))
        roi = (max(0, x1 - m), max(0, y1 - m), x2 + m, y2 + m)
    return chairs, desks, tvs, roi, max(len(chairs), len(desks), len(tvs)), items


def _draw_chair_area(img, rois, seat_total):
    for n, (x1r, y1r, x2r, y2r) in enumerate(rois, 1):
        cv2.rectangle(img, (x1r, y1r), (x2r, y2r), (0, 255, 255), 2)
        cv2.putText(img, f"SEAT AREA {n}", (x1r, y1r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def detect_prompt_seats(img, conf_thr=0.25, imgsz_val=640):
    """Detect every object type listed in the seat prompt -> [(box, label, conf)].
    Each detection counts as ONE seat. Runs sensitive: conf floor 0.25 and a
    fixed 640 px inference size so small/distant objects still fire."""
    if not live_prompt_hits:
        return []
    res = get_model(CHAIR_MODEL)(img, conf=min(conf_thr, 0.25),
                                 classes=[cid for cid, _ in live_prompt_hits],
                                 imgsz=640, verbose=False)[0]
    label_of = dict(live_prompt_hits)
    return [(b.xyxy[0].cpu().numpy(), label_of.get(int(b.cls), "seat"),
             float(b.conf))
            for b in res.boxes]


def _draw_prompt_seats(img, prompt_boxes):
    """Orange box per prompt-detected seat object."""
    for n, (box, label, _) in enumerate(prompt_boxes, 1):
        x1, y1, x2, y2 = np.asarray(box, dtype=int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(img, f"{label.upper()} #{n}", (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)


def _rects_from_canvas(result, scale):
    """Convert seatdraw result ({"objects":[{left,top,width,height}...]})
    -> list of (x1,y1,x2,y2) in original image pixels."""
    if not result or not result.get("objects"):
        return []
    rois = []
    for obj in result["objects"]:
        rois.append((int(obj["left"] * scale), int(obj["top"] * scale),
                     int((obj["left"] + obj["width"]) * scale),
                     int((obj["top"] + obj["height"]) * scale)))
    return rois


def _data_url(bgr):
    """Encode a BGR image as a JPEG base64 data URL (for the seatdraw component)."""
    import io as _io
    import base64 as _base64
    buf = _io.BytesIO()
    Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).convert("RGB") \
        .save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + _base64.b64encode(buf.getvalue()).decode()


def _save_training_frame(img, tag, idx, with_labels=True):
    """Break a video frame out into the dataset: image -> dataset/images/,
    auto-labels (person boxes from that frame's detection) -> dataset/labels/.
    verify_yolo.ipynb section 3 then trains class-monitor-ai on them.
    With with_labels=False the frame is stored as a PLAIN screenshot in
    dataset/plain_screenshots/ (kept OUT of the YOLO train folder - it has
    no .txt sidecar and must never break the label pairing)."""
    img_dir = os.path.join("dataset",
                           "plain_screenshots" if not with_labels else "images")
    lab_dir = os.path.join("dataset", "labels")
    os.makedirs(img_dir, exist_ok=True)
    stem = f"vid_{tag}_{idx:04d}"
    out_path = os.path.join(img_dir, stem + ".jpg")
    k = 1                                    # never overwrite an existing frame
    while os.path.exists(out_path):
        stem = f"vid_{tag}_{idx:04d}_v{k}"
        out_path = os.path.join(img_dir, stem + ".jpg")
        k += 1
    cv2.imwrite(out_path, img,
                [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not with_labels:
        return stem
    os.makedirs(lab_dir, exist_ok=True)
    lines = []
    res = _LAST_DET.get("res")
    h, w = img.shape[:2]
    min_conf = max(0.45, float(conf))
    if res is not None and getattr(res, "boxes", None) is not None:
        for b in res.boxes:
            if int(b.cls) != PERSON_ID or float(b.conf) < min_conf:
                continue
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f}"
                         f" {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    with open(os.path.join(lab_dir, stem + ".txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return stem


def _append_csv_rows(csv_path, image_file, src_idx, img_shape):
    """Append every item detected in one extracted frame to the run's CSV log
    (person + chair + desk + monitor). Human-review format; the YOLO .txt
    labels stay person-only. Returns the number of rows written."""
    h, w = img_shape[:2]
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    new_file = not os.path.exists(csv_path)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    res = _LAST_DET.get("res")
    min_conf = max(0.45, float(conf))
    if res is not None and getattr(res, "boxes", None) is not None:
        for b in res.boxes:
            if int(b.cls) == PERSON_ID and float(b.conf) >= min_conf:
                rows.append(("person", float(b.conf), b.xyxy[0].tolist()))
    rows.extend((name, cfx, [float(v) for v in box])
                for name, box, cfx in (_LAST_DET.get("items") or []))
    rows.extend((label.lower(), None, [float(v) for v in box])
                for box, label, _ in (_LAST_DET.get("prompt") or []))
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        if new_file:
            wr.writerow(["image_file", "source_frame", "class_name", "confidence",
                         "x_min", "y_min", "x_max", "y_max",
                         "img_w", "img_h", "saved_time"])
        for name, cfx, (x1, y1, x2, y2) in rows:
            wr.writerow([image_file, src_idx, name,
                         f"{cfx:.3f}" if cfx is not None else "",
                         int(x1), int(y1), int(x2), int(y2), w, h, stamp])
    return len(rows)


def analyze_image(img, conf_thr, seats_override, imgsz_val, manual_rois=None):
    """Run the full pipeline on ONE BGR image -> (annotated_img, occupied, seats).
    manual_rois: list of drawn (x1,y1,x2,y2) boxes overriding everything.
    Furniture (chair/desk/MONITOR) is ALWAYS scanned and tagged by name.
    Seat source priority: drawn boxes > seat-prompt detections > auto
    chair/desk/monitor union. NO fallback seat area: when nothing is
    recognized nothing is counted (red hint instead of a full-frame box)."""
    chairs, desks, tvs, auto_roi, furn, furn_items = _furniture_area(img)
    prompt_boxes = detect_prompt_seats(img, conf_thr)

    if manual_rois:
        rois = manual_rois
        zone_chrome = True                     # drawn boxes ARE the zones
        seat_total = seats_override if seats_override else len(manual_rois)
    elif prompt_boxes:
        rois = [tuple(map(int, b)) for b, _, _ in prompt_boxes]
        zone_chrome = False                    # named items -> no SEAT AREA overlay
        seat_total = seats_override if seats_override else len(rois)
    elif auto_roi is not None:
        rois = [auto_roi]
        zone_chrome = True                     # union zone around all furniture
        seat_total = seats_override if seats_override else furn
    else:
        rois = []                           # nothing recognized -> no seat area
        zone_chrome = False
        seat_total = seats_override if seats_override else 0

    results = get_model(model_name)(img, conf=conf_thr, imgsz=imgsz_val, verbose=False)[0]
    _LAST_DET.update(res=results, chairs=chairs, desks=desks,
                     tvs=tvs, items=furn_items, prompt=prompt_boxes)
    occupied = _draw_people(img, results, rois, conf_thr)
    _draw_furniture(img, chairs, desks, tvs)
    _draw_prompt_seats(img, prompt_boxes)
    if zone_chrome:
        _draw_chair_area(img, rois, seat_total)
    if not rois:
        cv2.putText(img, "NO SEAT OBJECTS FOUND", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    _draw_status(img, seat_total, occupied)
    return img, occupied, seat_total


def _draw_status(img, seats, occupied, fps=None):
    color = (0, 255, 0) if occupied < seats else (0, 0, 255)
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (40, 40, 40), -1)
    cv2.putText(img, f"SEATS: {seats}   OCCUPIED: {occupied}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if fps is not None:
        cv2.putText(img, f"FPS: {fps:.0f}", (img.shape[1] - 90, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


# --------------------------------------------------------------------------
# Video processor
# --------------------------------------------------------------------------
class OccupancyProcessor(VideoProcessorBase):
    MIN_INTERVAL = 0.06          # seconds between processed frames (~16 fps max, GPU-fast)
    CHAIR_EVERY = 2.0            # refresh the chairs every 2 seconds

    def __init__(self):
        self.fps = 0.0
        self._prev = time.time()
        self._last_process = 0.0
        self._last_chair = 0.0
        self._last_prompt = 0.0
        self._last_annotated = None
        self.auto_roi = None
        self.rois = []
        self.zone_chrome = True
        self.total_seats = 0
        self._furn_counts = []
        self._chair_boxes = np.empty((0, 4), dtype=float)
        self._desk_boxes = np.empty((0, 4), dtype=float)
        self._tv_boxes = np.empty((0, 4), dtype=float)
        self._prompt_boxes = []

    def _update_chair_area(self, chairs, desks, tvs):
        if live_manual_rois:                # manual boxes drawn -> keep them
            return
        all_boxes = list(chairs) + list(desks) + list(tvs)
        if not all_boxes:
            return
        arr = np.array(all_boxes)
        x1 = max(0, int(arr[:, 0].min()))
        y1 = max(0, int(arr[:, 1].min()))
        x2 = int(arr[:, 2].max())
        y2 = int(arr[:, 3].max())
        m = max(8, int(0.02 * (x2 - x1)))
        cur = (max(0, x1 - m), max(0, y1 - m), x2 + m, y2 + m)
        if self.auto_roi is None:
            self.auto_roi = cur
        else:
            a = 0.35
            self.auto_roi = tuple(int(a * n + (1 - a) * o)
                                  for n, o in zip(cur, self.auto_roi))
        self._furn_counts.append(max(len(chairs), len(desks), len(tvs)))
        self._furn_counts = self._furn_counts[-20:]
        if seats == 0:
            self.total_seats = int(np.median(self._furn_counts))

    def _detect_furniture(self, img):
        res = get_model(CHAIR_MODEL)(img, conf=FURN_CONF,
                                     classes=[CHAIR_ID, DESK_ID, TV_ID],
                                     imgsz=FURN_IMGSZ, verbose=False)[0]
        chairs = [b.xyxy[0].cpu().numpy() for b in res.boxes if int(b.cls) == CHAIR_ID]
        desks = [b.xyxy[0].cpu().numpy() for b in res.boxes if int(b.cls) == DESK_ID]
        tvs = [b.xyxy[0].cpu().numpy() for b in res.boxes if int(b.cls) == TV_ID]
        self._chair_boxes = np.array(chairs).reshape(-1, 4)
        self._desk_boxes = np.array(desks).reshape(-1, 4)
        self._tv_boxes = np.array(tvs).reshape(-1, 4)
        self._update_chair_area(chairs, desks, tvs)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        now = time.time()
        if now - self._last_process < self.MIN_INTERVAL:
            if self._last_annotated is not None:
                return av.VideoFrame.from_ndarray(self._last_annotated, format="bgr24")
            return frame

        try:
            with _INFER_LOCK:
                return self._process(frame, now)
        except Exception:
            traceback.print_exc()
            return frame

    def _process(self, frame: av.VideoFrame, now: float) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        _LIVE["frame"] = img
        if live_prompt_hits and now - self._last_prompt > self.CHAIR_EVERY:
            self._last_prompt = now
            self._prompt_boxes = detect_prompt_seats(img, max(0.2, conf - 0.1), 640)
        if now - self._last_chair > self.CHAIR_EVERY:   # furniture ALWAYS scanned
            self._last_chair = now
            self._detect_furniture(img)

        results = get_model(model_name)(img, conf=conf, imgsz=imgsz, verbose=False)[0]
        if live_manual_rois:                    # drawn boxes override auto
            if live_box_wh and (w, h) != tuple(live_box_wh):
                sw, sh = w / live_box_wh[0], h / live_box_wh[1]
                self.rois = [(int(x1 * sw), int(y1 * sh),
                              int(x2 * sw), int(y2 * sh))
                             for (x1, y1, x2, y2) in live_manual_rois]
            else:
                self.rois = list(live_manual_rois)
            self.zone_chrome = True
            if seats == 0:                      # each drawn box = one seat
                self.total_seats = len(live_manual_rois)
        elif self._prompt_boxes:                # prompt objects = one seat each
            self.rois = [tuple(map(int, b)) for b, _, _ in self._prompt_boxes]
            self.zone_chrome = False            # named items -> no SEAT AREA overlay
            if seats == 0:
                self.total_seats = len(self.rois)
        elif self.auto_roi is not None:
            self.rois = [self.auto_roi]         # union zone around all furniture
            self.zone_chrome = True
        else:
            self.rois = []                      # no seat area yet
            self.zone_chrome = False

        occupied = _draw_people(img, results, self.rois, conf)
        _draw_furniture(img, self._chair_boxes, self._desk_boxes, self._tv_boxes)
        _draw_prompt_seats(img, self._prompt_boxes)
        if self.zone_chrome:
            _draw_chair_area(img, self.rois, self.total_seats)
        if not self.rois:
            cv2.putText(img, "NO SEAT OBJECTS FOUND", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        now = time.time()
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(1e-6, now - self._prev))
        self._prev = now
        _draw_status(img, self.total_seats, occupied, self.fps)

        self._last_process = now
        self._last_annotated = img
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --------------------------------------------------------------------------
# WebRTC stream (browser camera -> YOLO -> annotated feed)
# --------------------------------------------------------------------------
ctx = webrtc_streamer(
    key="classroom-occupancy",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=OccupancyProcessor,
)

if not ctx.state.playing:
    st.info("👆 Press **START** and allow camera access on this device.")
else:
    st.success("Live counting... point the camera at the classroom chairs.")


# --------------------------------------------------------------------------
# Image upload (test with a static photo)
# --------------------------------------------------------------------------
st.divider()
st.subheader("Test with an uploaded image")
st.caption("Draw a box over the seat area with the canvas (or leave it empty for auto), "
           "then the pipeline counts heads inside it.")

up = st.file_uploader("Upload a classroom photo (JPG / PNG)", type=["jpg", "jpeg", "png"])
if up is not None:
    data = np.frombuffer(up.getvalue(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Could not decode that image. Try a JPG or PNG.")
    else:
        try:
            disp_w = 900
            scale = img.shape[1] / disp_w
            disp = cv2.resize(img, (disp_w, int(disp_w * img.shape[0] / img.shape[1])))
            disp_h = disp.shape[0]
            result = seatdraw(image_url=_data_url(disp), rects=[], height=disp_h,
                              key=f"seat-upload-{up.name}")
            manual_rois = _rects_from_canvas(result, scale)

            if st.button("Run detection"):
                annotated, occupied, seat_total = analyze_image(img, conf, seats, imgsz, manual_rois)
                c1, c2, c3 = st.columns(3)
                c1.metric("Seats", seat_total)
                c2.metric("Occupied", occupied)
                c3.metric("Free seats", max(0, seat_total - occupied))
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                         caption="Detection result", use_container_width=True)
            else:
                st.info("Draw seat boxes (optional), then press Run detection. "
                        "Leave empty for auto.")
        except Exception:
            traceback.print_exc()
            st.error("Analysis failed - see the terminal for the traceback.")


# --------------------------------------------------------------------------
# Video upload (test with a recorded clip: MP4 / MOV)
# --------------------------------------------------------------------------
st.divider()
st.subheader("Test with an uploaded video")
st.caption("MP4 / MOV / AVI. Every frame runs through the same pipeline "
           "(seat-prompt objects or the auto chair/desk/MONITOR area -> "
           "heads inside = occupied).")

up_vid = st.file_uploader("Upload a classroom video",
                          type=["mp4", "mov", "avi", "m4v"])
if up_vid is not None:
    suffix = os.path.splitext(up_vid.name)[1] or ".mp4"
    tmp_in = tempfile.NamedTemporaryFile(prefix="occ_", suffix=suffix, delete=False)
    while True:                     # 1 MB chunks - never copies the whole video
        chunk = up_vid.read(1024 * 1024)
        if not chunk:
            break
        tmp_in.write(chunk)
    tmp_in.close()

    cap = cv2.VideoCapture(tmp_in.name)
    if not cap.isOpened():
        st.error("Could not open that video - try MP4/H.264 or a normal MOV.")
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps <= 1:
            fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dur = total_frames / fps if fps else 0
        st.write(f"**{os.path.basename(up_vid.name)}** - {vw}x{vh}, "
                 f"{total_frames} frames @ {fps:.0f} fps ({dur:.0f}s)")

        secs = st.number_input("Seconds to process (0 = whole video)", 0, 600, 30,
                               help="Long videos take long to analyze - start with 30 s.")
        extract_cb = st.checkbox("Break video into training frames", value=True,
                                 help="Scans the WHOLE video and saves one frame every N "
                                      "seconds into dataset/images/ with YOLO person "
                                      "labels in dataset/labels/ and a full item log "
                                      "(person/chair/desk/monitor) in dataset/csv/. "
                                      "Train on them later in verify_yolo.ipynb "
                                      "(section 3).")
        every_s = st.number_input("Save a frame every ... seconds", 1, 30, 2)
        plain_cb = st.checkbox(
            "Plain screenshots - no AI boxes or labels",
            help="Saves the extracted frames exactly as the camera saw them "
                 "into dataset/plain_screenshots/ - nothing detected, "
                 "highlighted or labeled, no .txt / CSV. Kept out of the "
                 "training folder so YOLO never sees unlabeled images. Label "
                 "them by hand, then move them over to train.")
        if st.button("Process video"):
            max_frames = int(fps * secs) if secs else total_frames
            stride = max(1, int(round(fps * every_s)))
            vscale = 1280 / vw if vw > 1280 else 1.0
            ph = st.empty()
            bar = st.progress(0.0, text="Processing...")
            m1, m2, m3, m4 = st.columns(4)
            occs, seat_final, i = [], 0, 0
            try:
                while i < max_frames:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if vscale < 1.0:
                        frame = cv2.resize(frame, (1280, int(vh * vscale)))
                    annotated, occupied, seat_final = analyze_image(
                        frame, conf, seats, imgsz, manual_rois=None)
                    occs.append(occupied)
                    ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                             caption=f"frame {i + 1}/{max_frames} | SEATS {seat_final}"
                                     f" OCCUPIED {occupied}",
                             use_container_width=True)
                    bar.progress(min(1.0, (i + 1) / max(1, max_frames)),
                                 text=f"frame {i + 1}/{max_frames}")
                    annotated = None          # dispose this frame immediately
                    frame = None
                    if (i + 1) % 25 == 0:     # periodic RAM/GPU cache trim
                        gc.collect()
                        try:
                            import torch as _t
                            _t.cuda.empty_cache()
                        except Exception:
                            pass
                    i += 1
            except Exception:
                traceback.print_exc()
                st.error("Video analysis failed - see the terminal for the traceback.")
            finally:
                cap.release()

            bar.progress(1.0, text="Done.")
            m1.metric("Frames analyzed", len(occs))
            m2.metric("Seats", seat_final)
            m3.metric("Avg occupied", round(sum(occs) / len(occs), 1) if occs else 0)
            m4.metric("Peak occupied", max(occs) if occs else 0)

            # ---- whole-video frame extraction (independent of processing limit) ----
            if extract_cb:
                name_part = re.sub(r"[^A-Za-z0-9_-]", "",
                                   os.path.splitext(up_vid.name)[0])[:24] or "clip"
                run_tag = f"{name_part}_{time.strftime('%Y%m%d_%H%M%S')}"   # unique per run
                csv_path = os.path.join("dataset", "csv", f"vid_{run_tag}.csv")
                ex_bar = st.progress(0.0, text="Extracting training frames...")
                ex_ph = st.empty()
                cap2 = cv2.VideoCapture(tmp_in.name)
                n_saved, n_rows, idx = 0, 0, 0
                show = None
                denom = total_frames or max_frames or 1
                try:
                    while True:
                        ok, frame = cap2.read()
                        if not ok:
                            break
                        if idx % stride == 0:
                            if vscale < 1.0:
                                frame = cv2.resize(frame, (1280, int(vh * vscale)))
                            if plain_cb:            # raw screenshot - no AI pass
                                stem = _save_training_frame(
                                    frame, run_tag, n_saved, with_labels=False)
                                show = frame
                            else:
                                annotated, _, _ = analyze_image(
                                    frame, conf, seats, imgsz, manual_rois=None)
                                stem = _save_training_frame(frame, run_tag, n_saved)
                                n_rows += _append_csv_rows(csv_path, stem + ".jpg",
                                                           idx, frame.shape)
                                show = annotated
                            n_saved += 1
                            ex_ph.image(cv2.cvtColor(show, cv2.COLOR_BGR2RGB),
                                        caption=f"saved {n_saved} | source frame {idx}",
                                        use_container_width=True)
                            ex_bar.progress(min(1.0, idx / denom),
                                            text=f"frame {idx}/{denom} - saved {n_saved}")
                            annotated = None        # save-1 dispose-1: nothing
                            show = None             # accumulates across frames
                            frame = None
                            _LAST_DET.update(res=None, items=None, prompt=None)
                            if n_saved % 20 == 0:   # periodic RAM/GPU cache trim
                                gc.collect()
                                try:
                                    import torch as _t
                                    _t.cuda.empty_cache()
                                except Exception:
                                    pass
                        idx += 1
                except Exception:
                    traceback.print_exc()
                    st.error("Frame extraction failed - see the terminal.")
                finally:
                    cap2.release()
                ex_bar.progress(1.0, text="Extraction done.")
                if plain_cb:
                    st.success(f"Saved {n_saved} plain screenshot(s) to "
                               f"dataset/plain_screenshots/ (names start with "
                               f"'vid_{run_tag}', no labels or CSV created). Label "
                               f"them by hand and move them to dataset/images/ + "
                               f"labels/ before training.")
                else:
                    st.success(f"Saved {n_saved} labeled frame(s) to dataset/images/ "
                               f"(YOLO .txt labels in dataset/labels/, item log with "
                               f"{n_rows} rows in dataset/csv/vid_{run_tag}.csv). "
                               f"Names start with 'vid_{run_tag}'. Open "
                               f"verify_yolo.ipynb -> section 3 to train "
                               f"class-monitor-ai on them.")

            os.unlink(tmp_in.name)


# --------------------------------------------------------------------------
# Draw seat-area boxes for the LIVE feed (reliable camera-photo approach)
# --------------------------------------------------------------------------
st.divider()
st.subheader("Draw seat-area boxes")
st.caption("Take a photo, then drag as many boxes as you need over the seats. "
           "Each box is applied to the live feed automatically - heads inside ANY "
           "box count as occupied. Draw / Select / resize / delete with the buttons.")

photo = st.camera_input("Take a photo to draw the seat area")
if photo is not None:
    data = np.frombuffer(photo.getvalue(), np.uint8)
    snap = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if snap is not None:
        disp_w = 900
        scale = snap.shape[1] / disp_w
        disp = cv2.resize(snap, (disp_w, int(disp_w * snap.shape[0] / snap.shape[1])))
        disp_h = disp.shape[0]
        result = seatdraw(image_url=_data_url(disp), rects=[], height=disp_h,
                          key=f"seat-setup-canvas-{st.session_state.get('seat_key', 0)}")
        boxes = _rects_from_canvas(result, scale)
        if boxes:                       # boxes apply automatically as you draw
            st.session_state["seat_boxes"] = boxes
            st.session_state["seat_boxes_wh"] = (int(snap.shape[1]), int(snap.shape[0]))

if st.session_state.get("seat_boxes"):
    n = len(st.session_state["seat_boxes"])
    st.success(f"{n} seat box(es) active on the live feed - heads inside ANY box "
               "count as occupied.")
    if st.button("Clear seat boxes"):
        st.session_state["seat_boxes"] = []
        st.session_state["seat_boxes_wh"] = None
        st.session_state["seat_key"] = st.session_state.get("seat_key", 0) + 1
        st.rerun()
else:
    st.info("No seat boxes yet - take a photo above and draw them. "
            "(Leave empty to use auto-detection.)")
