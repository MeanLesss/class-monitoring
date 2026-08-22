# Classroom Occupancy

YOLO-based live people counter for a classroom. It tracks each person's
**upper body and head (facial tracking via pose keypoints)** and counts a seat as
occupied only when the person's **head is inside the chair area**.

## How it works

1. **Seats** — the app detects **chairs (class 56) AND desks (class 60, dining
   table as proxy)** in the same pass. Desks are used because sitting people can
   hide the chairs. `SEATS = median of max(chairs, desks)` (override with `--seats`).
2. **Static seat box** — one yellow box is locked around the chair+desk area.
3. **Upper body + head** — a pose model finds each person; from the face
   keypoints (nose/eyes/ears) it builds a HEAD box (cyan, "FACE" label) and an
   UPPER-BODY box (green/grey). No full-body box is drawn.
4. **Count occupancy** — a person counts as occupied only when their **head
   center** is inside the seat-area box. Heads outside (e.g. standing by the
   door) are grey and not counted.

## Files

| File | What it does |
|------|--------------|
| `detect_live.py` | Live counter on a **PC webcam** (or video file) |
| `app.py` | Live counter in the **browser / phone** (Streamlit + WebRTC) |
| `collect_dataset.py` | Capture 100–200 classroom photos for the dataset |
| `verify_yolo.ipynb` | Notebook: webcam + image picker + verify pretrained YOLO + find chairs + fine-tune |
| `custom.yaml` | YOLO data config used for fine-tuning |
| `dataset/images` | Your classroom photos (occupied / empty) |
| `dataset/labels` | Optional YOLO-format labels (class 0 = person) |
| `dataset/outputs` | Annotated photos written by the notebook |
| `models/pretrained` | Pretrained YOLO weights (`yolo11n.pt`, `yolo11n-pose.pt`, `yolov8n*.pt`) |
| `models/tuned` | Fine-tuned weights from the notebook training step |

## Setup (once)

```powershell
python -m venv ai-venv
.\ai-venv\Scripts\pip install -r requirements.txt
```

To use the **NVIDIA GPU** (training + live detection run much faster), install the
CUDA build of torch instead of the CPU-only default:

```powershell
.\ai-venv\Scripts\pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Check it: `python -c "import torch; print(torch.cuda.is_available())"` → `True`.

(On Colab, `!pip install ultralytics` and upload the notebook. Colab already has a GPU.)

## 1. Live counter — PC webcam

```powershell
.\ai-venv\Scripts\python.exe detect_live.py
.\ai-venv\Scripts\python.exe detect_live.py --model models\pretrained\yolov8n-pose.pt
.\ai-venv\Scripts\python.exe detect_live.py --seats 30        # force seat total
.\ai-venv\Scripts\python.exe detect_live.py --roi 100,150,900,600
.\ai-venv\Scripts\python.exe detect_live.py --select-roi      # draw the box by hand
```

- Yellow box = seat area, cyan box = head ("FACE"), green upper-body = counted,
  grey = ignored, orange = chair, magenta = desk. Status bar:
  `SEATS: N   OCCUPIED: K` (red when full). `ESC` exits.
- Speed: lower inference size = higher FPS. Try `--imgsz 416` (faster) or
  `--imgsz 640` (more detail).

## 2. Live counter — browser / phone (no PC camera needed)

```powershell
.\ai-venv\Scripts\streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Open `http://<PC_IP>:8501` on the phone (same Wi-Fi), press **START**, allow the
camera, and point it at the chairs. The app detects the chairs+desks, locks the
seat area, and counts only heads inside it. Sidebar: person model, confidence,
seat override, inference size, and a **manual seat-area box** (Left/Top/Right/
Bottom %) to override the auto-detected area live.

**Drawing the seat boxes with the cursor:**
- **Live feed**: take a photo with the **"Take a photo to draw the seat area"**
  widget, then drag as many boxes as you need over the seats. Each box applies to
  the live feed automatically — heads inside ANY box count as occupied. Use the
  Draw / Select buttons to add, move, resize, or delete boxes. **Clear seat boxes**
  resets to auto.
- **Uploaded image**: same drawing on the photo, then **Run detection**.

## 3. Collect the dataset (100–200 photos)

```powershell
.\ai-venv\Scripts\python.exe collect_dataset.py --count 150
```

`SPACE` saves a photo to `dataset/images/`, `R` hints a new camera angle
(high / eye-level / side, occupied vs empty), `ESC` exits.

## 4. Notebook — verify, capture, find chairs, train

Open `verify_yolo.ipynb` (VS Code, Jupyter, or Colab). It:

1. **Adds photos** — requests the webcam (local camera first, then the browser's);
   if no camera is available it shows an **image picker**.
2. **Verifies pretrained YOLO** — runs `yolov8n/s`, `yolo11n`, `*-pose` over every
   photo, compares counts/confidences, saves annotated outputs.
3. **Finds the chairs** — counts chairs per photo, draws the chair-area box, and
   checks chair-aware occupancy (people inside the box).
4. **Trains (image loop)** — auto-labels any unlabeled photo, splits train/val,
   and fine-tunes `yolo11n.pt` so the model learns *your* seats and angles.
5. Saves tuned weights to `models/tuned/weights/best.pt`.

Use the tuned model live:

```powershell
.\ai-venv\Scripts\python.exe detect_live.py --model models\tuned\weights\best.pt
```
