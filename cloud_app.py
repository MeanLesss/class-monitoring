"""
Classroom Occupancy - Streamlit Community Cloud edition.

Same detection pipeline as app.py (YOLO pose -> head/upper-body boxes, chairs
detected as the seat area). Detection logic:
- A POSE model finds each person and, from the face keypoints (nose/eyes/ears),
  builds a HEAD box and an UPPER-BODY box (no full-body box).
- Chairs (COCO class 56) are detected with a light detect model every ~2 s.
- A person counts as OCCUPIED only if their HEAD center is inside the seat area.

Input options (combined from app.py + cloud-friendly fallbacks):
1. LIVE camera via streamlit-camera-input-live (no WebRTC / no TURN relay, so it
   works reliably on the free Streamlit Cloud; a few frames/sec on CPU).
2. On-demand camera capture via st.camera_input (always works on the cloud).
3. Uploaded image / video (always works on the cloud).

Local run (from this folder):
    .\\ai-venv\\Scripts\\streamlit run cloud_app.py
"""

import csv
import gc
import json
import os
import re
import tempfile
import traceback
import urllib.request

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from seatdraw import seatdraw

from ultralytics import YOLO

# Live camera: we use streamlit-camera-input-live, which streams the browser
# webcam as JPEG frames over Streamlit's normal component channel. Unlike
# streamlit-webrtc it does NOT use WebRTC/P2P, so it needs no TURN relay and
# works reliably on the free Streamlit Cloud. It is lower-FPS (a few frames/s)
# but hosted-friendly. If the package is missing the rest of the app still runs.
try:
    from camera_input_live import camera_input_live
    _HAVE_LIVE = True
except Exception:  # pragma: no cover - live camera is optional
    camera_input_live = None
    _HAVE_LIVE = False

# --------------------------------------------------------------------------
# Model paths. On the cloud the pretrained weights are NOT stored in git, so
# they're downloaded on first run to a local cache dir. ultralytics >= 8.3
# has a built-in asset cache (~/.cache/ultralytics) and can auto-fetch by name,
# but we download explicitly so we control where they land.
# --------------------------------------------------------------------------
MODEL_BASE = os.environ.get("MODEL_BASE", "https://github.com/ultralytics/assets/releases/download/v8.3.0")

# Local storage dir where downloaded weights are cached (survives across reruns).
_CACHE = os.path.join(os.environ.get("STREAMLIT_CACHE", tempfile.gettempdir()), "class-monitor-models")
os.makedirs(_CACHE, exist_ok=True)

CHAIR_MODEL = "models/pretrained/yolo11n.pt"
TRAINED_MODEL = "models/tuned/class-monitor-ai.pt"   # optional: your fine-tune

# Pretrained model files -> what should be in the git repo / downloaded.
_PRETRAINED_REPO = {
    "models/pretrained/yolo11n-pose.pt": ("yolo11n-pose.pt", MODEL_BASE),
    "models/pretrained/yolov8n-pose.pt": ("yolov8n-pose.pt", MODEL_BASE),
    "models/pretrained/yolo11n.pt":      ("yolo11n.pt",      MODEL_BASE),
    "models/pretrained/yolov8n.pt":      ("yolov8n.pt",      MODEL_BASE),
}

PERSON_ID = 0   # COCO class id for "person"
CHAIR_ID = 56   # COCO class id for "chair"
DESK_ID = 60    # COCO class id for "dining table" (used as the desk proxy)
TV_ID = 62      # COCO class id for "tv" (shown as MONITOR - no 'monitor' class exists)
FURN_CONF = 0.22    # low confidence floor for the furniture scan (small/distant items)
FURN_IMGSZ = 640    # fixed inference size for the furniture scan (finds small screens)

# Prompt words that are not exact COCO names get mapped to these
PROMPT_SYNONYMS = {
    "monitor": ["tv"], "screens": ["tv"], "screen": ["tv"],
    "computers": ["laptop", "tv"], "computer": ["laptop"], "pc": ["laptop"],
    "desk": ["dining table"], "desks": ["dining table"],
    "table": ["dining table"], "tables": ["dining table"],
    "sofa": ["couch"], "sofas": ["couch"],
    "phone": ["cell phone"], "phones": ["cell phone"],
}


def ensure_model(path):
    """Return absolute path to a model weight file, downloading it if needed.
    Prefers the local/git file; falls back to the cached/downloaded copy."""
    if os.path.exists(path):
        return os.path.abspath(path)
    if path not in _PRETRAINED_REPO:
        raise FileNotFoundError(f"Model {path!r} not found locally and has no "
                                "automatic download source.")
    fname, base = _PRETRAINED_REPO[path]
    dest = os.path.join(_CACHE, fname)
    if not os.path.exists(dest):
        url = f"{base}/{fname}"
        print(f"[class-monitor] downloading {fname} ...")
        urllib.request.urlretrieve(url, dest)
    return dest


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
        help="Pose models give upper-body + head/facial tracking.",
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
            st.warning(f"class-monitor-ai is weak (mAP50 "
                       f"{_map50 if _map50 is not None else 'n/a'}).")
        elif model_name == TRAINED_MODEL:
            st.caption(f"Your trained model - mAP50 {_map50:.2f}")
    else:
        _conf_floor = None
    conf = st.slider("Confidence", 0.1, 0.9, 0.4, 0.05)
    if (model_name == TRAINED_MODEL and _conf_floor
            and _map50 is not None and _map50 >= 0.3 and _conf_floor < conf):
        conf = _conf_floor
        st.caption(f"confidence auto-set to {conf} for your tuned model.")
    seats = st.number_input("Total seats (0 = auto from seats found)", 0, 200, 0)
    imgsz = st.select_slider("Speed (inference size)", [320, 416, 480, 640], 480,
                             help="Lower = faster, slightly less detail.")
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
        st.write("1. Finds the chairs and locks the seat area box.")
    st.write("2. Tracks each person's HEAD (nose/eyes/ears) + upper body.")
    st.write("3. Only heads inside a seat box count as occupied.")


# --------------------------------------------------------------------------
# Models - loaded EAGERLY in the main thread. We cache them in a module-level
# dict guarded by a lock (NOT @st.cache_resource) to avoid Streamlit's
# cache-replay mechanism, which errors on the cloud when a cached function
# emits Streamlit elements. Module globals persist across reruns in the same
# process, which is all we need.
# --------------------------------------------------------------------------
_MODEL_CACHE: dict = {}
_MODEL_LOCK = __import__("threading").Lock()


def _load_models(person_path, chair_path):
    with _MODEL_LOCK:
        person = _MODEL_CACHE.get(person_path)
        if person is None:
            person = YOLO(ensure_model(person_path))
            _MODEL_CACHE[person_path] = person
        chair = _MODEL_CACHE.get(chair_path)
        if chair is None:
            chair = YOLO(ensure_model(chair_path))
            _MODEL_CACHE[chair_path] = chair
    return person, chair


_person_model, _chair_model = _load_models(model_name, CHAIR_MODEL)

IS_POSE = "pose" in model_name.lower()


def _get_person():
    return _person_model


def _get_chair():
    return _chair_model


def _warmup(model):
    try:
        model.predict(np.zeros((240, 320, 3), dtype=np.uint8),
                      conf=0.9, imgsz=320, device="cpu", verbose=False)
    except Exception:
        pass


if "_WARMED" not in globals():
    _WARMED = True
    _warmup(_person_model)
    _warmup(_chair_model)


# --------------------------------------------------------------------------
# Seat-object prompt ("chair, monitor, laptop") -> COCO class ids
# --------------------------------------------------------------------------
_PROMPT_CACHE = {}


def resolve_seat_prompt(text):
    if not text or not text.strip():
        return [], []
    key = " ".join(text.lower().split())
    if key in _PROMPT_CACHE:
        return _PROMPT_CACHE[key]
    id_by_name = {str(v).lower(): int(k)
                  for k, v in _chair_model.names.items()}
    hits, missing, seen = [], [], set()
    for raw in re.split(r"[,;\n]+", text):
        w = raw.strip().lower()
        if not w:
            continue
        w2 = w[:-1] if w.endswith("s") and not w.endswith("ss") else w
        cands = [w, w2] + [s for k in (w, w2) for s in PROMPT_SYNONYMS.get(k, [])]
        hit = None
        for c in cands:
            if c and c in id_by_name:
                hit = (id_by_name[c], c)
                break
        if hit is None:
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
live_prompt_hits = seat_hits

if seat_missing:
    st.warning("Ignored (no such object class): "
               + ", ".join(seat_missing)
               + ". Try e.g. chair, tv, laptop, dining table, couch, keyboard.")
elif seat_hits:
    st.caption("Seat objects: **" + ", ".join(lbl for _, lbl in seat_hits) + "**")


def head_and_upper(kpts, conf_thr):
    pts = {i: (float(kpts[i][0]), float(kpts[i][1]))
           for i in range(17) if float(kpts[i][2]) > conf_thr}
    if 0 not in pts:
        return None, None, []
    face = [pts[i] for i in (0, 1, 2, 3, 4) if i in pts]
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


def _draw_people(img, results, rois, conf_thr):
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
    res = _get_chair()(img, conf=FURN_CONF,
                       classes=[CHAIR_ID, DESK_ID, TV_ID],
                       imgsz=FURN_IMGSZ, device="cpu", verbose=False)[0]
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
    if not live_prompt_hits:
        return []
    res = _get_chair()(img, conf=min(conf_thr, 0.25),
                       classes=[cid for cid, _ in live_prompt_hits],
                       imgsz=640, device="cpu", verbose=False)[0]
    label_of = dict(live_prompt_hits)
    return [(b.xyxy[0].cpu().numpy(), label_of.get(int(b.cls), "seat"),
             float(b.conf))
            for b in res.boxes]


def _draw_prompt_seats(img, prompt_boxes):
    for n, (box, label, _) in enumerate(prompt_boxes, 1):
        x1, y1, x2, y2 = np.asarray(box, dtype=int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(img, f"{label.upper()} #{n}", (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)


def _rects_from_canvas(result, scale):
    if not result or not result.get("objects"):
        return []
    rois = []
    for obj in result["objects"]:
        rois.append((int(obj["left"] * scale), int(obj["top"] * scale),
                     int((obj["left"] + obj["width"]) * scale),
                     int((obj["top"] + obj["height"]) * scale)))
    return rois


def _data_url(bgr):
    import io as _io
    import base64 as _base64
    buf = _io.BytesIO()
    Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).convert("RGB") \
        .save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + _base64.b64encode(buf.getvalue()).decode()


def _draw_status(img, seats_count, occupied, fps=None):
    color = (0, 255, 0) if occupied < seats_count else (0, 0, 255)
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (40, 40, 40), -1)
    cv2.putText(img, f"SEATS: {seats_count}   OCCUPIED: {occupied}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if fps is not None:
        cv2.putText(img, f"FPS: {fps:.0f}", (img.shape[1] - 90, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def analyze_image(img, conf_thr, seats_override, imgsz_val, manual_rois=None, results_out=None):
    chairs, desks, tvs, auto_roi, furn, furn_items = _furniture_area(img)
    prompt_boxes = detect_prompt_seats(img, conf_thr)

    if manual_rois:
        rois = manual_rois
        zone_chrome = True
        seat_total = seats_override if seats_override else len(manual_rois)
    elif prompt_boxes:
        rois = [tuple(map(int, b)) for b, _, _ in prompt_boxes]
        zone_chrome = False
        seat_total = seats_override if seats_override else len(rois)
    elif auto_roi is not None:
        rois = [auto_roi]
        zone_chrome = True
        seat_total = seats_override if seats_override else furn
    else:
        rois = []
        zone_chrome = False
        seat_total = seats_override if seats_override else 0

    results = _get_person()(img, conf=conf_thr, imgsz=imgsz_val,
                            device="cpu", verbose=False)[0]
    if results_out is not None:
        results_out["res"] = results
        results_out["items"] = furn_items
        results_out["prompt"] = prompt_boxes
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


def _render_analyze(img_bgr, key_tag, prefilled=None):
    """Shared runner: show a seatdraw canvas, then analyze on a button press.
    prefilled: optional list of (x1,y1,x2,y2) boxes (pixels) used in place of a
    fresh canvas (e.g. the saved seat-area boxes for the camera capture)."""
    try:
        manual_rois = None
        if prefilled:
            manual_rois = prefilled
        else:
            disp_w = 900
            scale = img_bgr.shape[1] / disp_w
            disp = cv2.resize(img_bgr,
                              (disp_w, int(disp_w * img_bgr.shape[0] / img_bgr.shape[1])))
            disp_h = disp.shape[0]
            result = seatdraw(image_url=_data_url(disp), rects=[], height=disp_h,
                              key=f"canvas-{key_tag}")
            manual_rois = _rects_from_canvas(result, scale)
        if st.button(f"Run detection **{key_tag}**"):
            annotated, occupied, seat_total = analyze_image(
                img_bgr, conf, seats, imgsz, manual_rois)
            c1, c2, c3 = st.columns(3)
            c1.metric("Seats", seat_total)
            c2.metric("Occupied", occupied)
            c3.metric("Free seats", max(0, seat_total - occupied))
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     caption="Detection result", width="stretch")
        else:
            st.info("Optional: draw seat boxes, then press Run detection. "
                    "Leave empty for auto.")
    except Exception:
        traceback.print_exc()
        st.error("Analysis failed - see logs for the traceback.")


# --------------------------------------------------------------------------
# 1. LIVE camera (camera_input_live) - no WebRTC, works on free Streamlit Cloud
# --------------------------------------------------------------------------
st.divider()
st.subheader("Live camera feed")
st.caption("Streams your browser webcam and runs YOLO live (a few frames/sec). "
           "Uses camera-input-live (not WebRTC), so it needs no TURN relay and "
           "works reliably on the free Streamlit Cloud. Untick to stop.")

live_manual_rois = list(st.session_state.get("seat_boxes", []))
live_prompt_hits = seat_hits

_live_on = st.checkbox("Start live detection", value=False)

if _HAVE_LIVE and _live_on:
    _live_ph = st.empty()
    try:
        _cframe = camera_input_live(debounce=300, key="live-cam",
                                    show_controls=True)
        if _cframe is None:
            st.info("Waiting for camera access - press Start capturing and allow "
                    "permission on this device.")
        else:
            _l_data = np.frombuffer(_cframe.getvalue(), np.uint8)
            _l_img = cv2.imdecode(_l_data, cv2.IMREAD_COLOR)
            if _l_img is not None:
                _annotated, _occ, _tot = analyze_image(
                    _l_img, conf, seats, imgsz,
                    manual_rois=live_manual_rois or None)
                _live_ph.image(cv2.cvtColor(_annotated, cv2.COLOR_BGR2RGB),
                               caption=f"SEATS {_tot}  OCCUPIED {_occ}",
                               width="stretch")
    except Exception:
        traceback.print_exc()
        st.error("Live detection failed - see logs for the traceback.")
elif not _HAVE_LIVE:
    st.warning("Live camera is disabled because `streamlit-camera-input-live` is "
               "not installed in this deployment. Use the on-demand capture / "
               "uploads below, or add it to requirements.txt.")
else:
    st.info("Unticked - live detection is off. Use the on-demand camera capture "
            "below for a single photo.")


# --------------------------------------------------------------------------
# 2. On-demand camera capture (cloud-friendly fallback - always works)
# --------------------------------------------------------------------------
st.header("Capture a live photo")
st.caption("Click below, the app asks for camera access, then press Run detection "
           "on the snapshot. Works on any device with a webcam - no live streaming "
           "needed (that's the cloud-friendly approach).")
_live_photo = st.camera_input("Take a photo to count occupancy")
if _live_photo is not None:
    _data = np.frombuffer(_live_photo.getvalue(), np.uint8)
    _img = cv2.imdecode(_data, cv2.IMREAD_COLOR)
    if _img is not None:
        st.image(_live_photo, caption="Captured photo", width="stretch")
        _render_analyze(_img, "camera",
                        prefilled=st.session_state.get("seat_boxes") or None)


# --------------------------------------------------------------------------
# 3. Image upload (test with a static photo)
# --------------------------------------------------------------------------
st.divider()
st.subheader("Test with an uploaded image")
st.caption("Draw a box over the seat area with the canvas (or leave it empty for "
           "auto), then the pipeline counts heads inside it.")

up = st.file_uploader("Upload a classroom photo (JPG / PNG)", type=["jpg", "jpeg", "png"])
if up is not None:
    data = np.frombuffer(up.getvalue(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Could not decode that image. Try a JPG or PNG.")
    else:
        _render_analyze(img, f"upload-{up.name}")


# --------------------------------------------------------------------------
# 4. Video upload (test with a recorded clip: MP4 / MOV)
# --------------------------------------------------------------------------
st.divider()
st.subheader("Test with an uploaded video")
st.caption("MP4 / MOV / AVI. Every frame runs through the same pipeline "
           "(seat-prompt objects or the auto chair/desk/MONITOR area -> "
           "heads inside = occupied). Keep clips short on the free cloud.")

up_vid = st.file_uploader("Upload a classroom video",
                          type=["mp4", "mov", "avi", "m4v"])
if up_vid is not None:
    suffix = os.path.splitext(up_vid.name)[1] or ".mp4"
    tmp_in = tempfile.NamedTemporaryFile(prefix="occ_", suffix=suffix, delete=False)
    while True:
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
        if st.button("Process video"):
            max_frames = int(fps * secs) if secs else total_frames
            vscale = 1280 / vw if vw > 1280 else 1.0
            ph = st.empty()
            bar = st.progress(0.0, text="Processing...")
            m1, m2, m3, m4 = st.columns(4)
            occs, seat_final, i = [], 0, 0
            results_out = {}
            try:
                while i < max_frames:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if vscale < 1.0:
                        frame = cv2.resize(frame, (1280, int(vh * vscale)))
                    annotated, occupied, seat_final = analyze_image(
                        frame, conf, seats, imgsz, manual_rois=None,
                        results_out=results_out)
                    occs.append(occupied)
                    ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                             caption=f"frame {i + 1}/{max_frames} | SEATS {seat_final}"
                                     f" OCCUPIED {occupied}",
                             width="stretch")
                    bar.progress(min(1.0, (i + 1) / max(1, max_frames)),
                                 text=f"frame {i + 1}/{max_frames}")
                    annotated = None
                    frame = None
                    if (i + 1) % 25 == 0:
                        gc.collect()
                    i += 1
            except Exception:
                traceback.print_exc()
                st.error("Video analysis failed - see logs for the traceback.")
            finally:
                cap.release()

            bar.progress(1.0, text="Done.")
            m1.metric("Frames analyzed", len(occs))
            m2.metric("Seats", seat_final)
            m3.metric("Avg occupied", round(sum(occs) / len(occs), 1) if occs else 0)
            m4.metric("Peak occupied", max(occs) if occs else 0)
            os.unlink(tmp_in.name)


# --------------------------------------------------------------------------
# 5. Draw seat-area boxes that apply to the camera capture (cloud equivalent of
#    the live seat-box drawing)
# --------------------------------------------------------------------------
st.divider()
st.subheader("Set seat-area boxes for the camera capture")
st.caption("Take a photo, then drag as many boxes as you need over the seats. "
           "Heads inside ANY box count as occupied. These apply when you press "
           "Run detection on the captured photo above.")

photo = st.camera_input("Take a photo to draw the seat area", key="seat_box_photo")
if photo is not None:
    data = np.frombuffer(photo.getvalue(), np.uint8)
    snap = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if snap is not None:
        disp_w = 900
        scale = snap.shape[1] / disp_w
        disp = cv2.resize(snap,
                          (disp_w, int(disp_w * snap.shape[0] / snap.shape[1])))
        disp_h = disp.shape[0]
        result = seatdraw(image_url=_data_url(disp), rects=[], height=disp_h,
                          key=f"seat-setup-canvas-{st.session_state.get('seat_key', 0)}")
        boxes = _rects_from_canvas(result, scale)
        if boxes:
            st.session_state["seat_boxes"] = boxes
            st.session_state["seat_boxes_wh"] = (int(snap.shape[1]), int(snap.shape[0]))
            st.session_state["seat_key"] = st.session_state.get("seat_key", 0) + 1
            st.rerun()

if st.session_state.get("seat_boxes"):
    n = len(st.session_state["seat_boxes"])
    st.success(f"{n} seat box(es) ready - the camera capture will use them.")
    if st.button("Clear seat boxes"):
        st.session_state["seat_boxes"] = []
        st.session_state["seat_boxes_wh"] = None
        st.session_state["seat_key"] = st.session_state.get("seat_key", 0) + 1
        st.rerun()
else:
    st.info("No seat boxes yet - take a photo above and draw them. "
            "(Leave empty to use auto-detection.)")
