"""
Classroom Occupancy - Live counter (upper-body + head/facial tracking).

Detection logic:
- A POSE model finds each person and, from the face keypoints (nose/eyes/ears),
  builds a HEAD box and an UPPER-BODY box - no full-body box.
- Chairs (COCO class 56) are detected with a light detect model, but only every
  few frames (they don't move), so it costs almost nothing.
- A person counts as OCCUPIED only if their HEAD center is inside the chair area.

Usage:
    python detect_live.py
    python detect_live.py --model models/pretrained/yolov8n-pose.pt
    python detect_live.py --seats 30
    python detect_live.py --roi 100,150,900,600
    python detect_live.py --select-roi
"""

import argparse
import time

import cv2
import numpy as np

from ultralytics import YOLO

PERSON_ID = 0    # COCO class id for "person"
CHAIR_ID = 56    # COCO class id for "chair"
DESK_ID = 60     # COCO class id for "dining table" (used as the desk proxy)


def head_and_upper(kpts, conf_thr):
    """From 17 pose keypoints build (head_box, upper_body_box, face_points).

    head = box around nose/eyes/ears (facial tracking).
    upper = box from the head down to the shoulders/hips (no legs).
    Returns (None, None, []) if the face is not visible.
    """
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

    body = [pts[i] for i in (0, 1, 2, 3, 4, 5, 6, 11, 12) if i in pts]  # head..shoulders..hips
    upper = (int(min(p[0] for p in body)), int(min(p[1] for p in body)),
             int(max(p[0] for p in body)), int(max(p[1] for p in body)))
    return head, upper, [(int(p[0]), int(p[1])) for p in face]


class ClassroomOccupancy:
    def __init__(self, model_path="models/pretrained/yolo11n-pose.pt", conf=0.4,
                 seats=0, roi=None, chair_model="models/pretrained/yolo11n.pt",
                 chair_conf=0.25, chair_every=20, select_roi=False,
                 imgsz=480, chair_imgsz=320):
        self.model = YOLO(model_path)
        self.chair_model = YOLO(chair_model)
        self.conf = conf
        self.chair_conf = chair_conf
        self.imgsz = imgsz               # lower = faster
        self.chair_imgsz = chair_imgsz
        self.is_pose = "pose" in model_path.lower()
        self.seats_override = seats
        self.total_seats = seats
        self.roi = tuple(roi) if roi else None
        self._roi_fixed = roi is not None
        self.select_roi = select_roi
        self.chair_every = chair_every          # update furniture only every N frames
        self._frame_idx = 0
        self._furn_counts = []
        self._chair_boxes = np.empty((0, 4), dtype=float)   # last detected chairs
        self._desk_boxes = np.empty((0, 4), dtype=float)    # last detected desks
        self.fps = 0.0

    # ------------------------------------------------------- furniture / seats
    def _update_chair_area(self, chairs, desks):
        """Stabilize the seat-area box from chairs AND desks (union).

        Desks are detected too because sitting people can cover the chairs;
        the desk surface usually stays visible, keeping the seat area stable.
        """
        all_boxes = list(chairs) + list(desks)
        if not all_boxes:
            return
        arr = np.array(all_boxes)
        x1 = max(0, int(arr[:, 0].min()))
        y1 = max(0, int(arr[:, 1].min()))
        x2 = int(arr[:, 2].max())
        y2 = int(arr[:, 3].max())
        m = max(8, int(0.02 * (x2 - x1)))
        cur = (max(0, x1 - m), max(0, y1 - m), x2 + m, y2 + m)
        if self.roi is None:
            self.roi = cur
        elif not self._roi_fixed:
            a = 0.35
            self.roi = tuple(int(a * n + (1 - a) * o) for n, o in zip(cur, self.roi))
        self._furn_counts.append(max(len(chairs), len(desks)))
        self._furn_counts = self._furn_counts[-20:]
        if not self.seats_override and self._furn_counts:
            self.total_seats = int(np.median(self._furn_counts))

    def _detect_furniture(self, frame):
        res = self.chair_model(frame, conf=self.chair_conf,
                               classes=[CHAIR_ID, DESK_ID], verbose=False,
                               imgsz=self.chair_imgsz)[0]
        chairs = [b.xyxy[0].cpu().numpy() for b in res.boxes if int(b.cls) == CHAIR_ID]
        desks = [b.xyxy[0].cpu().numpy() for b in res.boxes if int(b.cls) == DESK_ID]
        self._chair_boxes = np.array(chairs).reshape(-1, 4)
        self._desk_boxes = np.array(desks).reshape(-1, 4)
        self._update_chair_area(chairs, desks)

    # ----------------------------------------------------------------- per frame
    def process(self, frame):
        self._frame_idx += 1
        if self._frame_idx % self.chair_every == 0:
            self._detect_furniture(frame)
        if self.roi is None:
            self.roi = (0, 0, frame.shape[1] - 1, frame.shape[0] - 1)

        results = self.model(frame, conf=self.conf, verbose=False, imgsz=self.imgsz)[0]
        keyp = results.keypoints
        x1r, y1r, x2r, y2r = self.roi
        occupied = 0

        for i, b in enumerate(results.boxes):
            if int(b.cls) != PERSON_ID:
                continue
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)

            head, upper, face = (None, None, [])
            if self.is_pose and keyp is not None and len(keyp) > i:
                head, upper, face = head_and_upper(keyp.data[i].cpu().numpy(), self.conf)
            if upper is None:                       # face not visible -> fallback
                upper = (x1, y1, x2, int(y1 + 0.6 * (y2 - y1)))
            if head is None:
                head = (x1, y1, x2, int(y1 + 0.3 * (y2 - y1)))

            hcx, hcy = (head[0] + head[2]) / 2, (head[1] + head[3]) / 2
            inside = x1r <= hcx <= x2r and y1r <= hcy <= y2r
            color = (0, 255, 0) if inside else (140, 140, 140)

            cv2.rectangle(frame, (upper[0], upper[1]), (upper[2], upper[3]), color, 2)
            cv2.putText(frame, f"p{float(b.conf):.2f}", (upper[0], upper[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # head box + facial tracking
            cv2.rectangle(frame, (head[0], head[1]), (head[2], head[3]),
                          (255, 255, 0), 2)
            for fx, fy in face:
                cv2.circle(frame, (fx, fy), 2, (0, 255, 255), -1)
            if face:
                cv2.putText(frame, "FACE", (head[0], head[3] + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            if inside:
                occupied += 1

        for cx1, cy1, cx2, cy2 in self._chair_boxes.astype(int):
            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 170, 0), 1)
        for dx1, dy1, dx2, dy2 in self._desk_boxes.astype(int):
            cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (255, 0, 255), 1)
            cv2.putText(frame, "DESK", (dx1, dy2 + 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 0, 255), 1)

        cv2.rectangle(frame, (x1r, y1r), (x2r, y2r), (0, 255, 255), 2)
        cv2.putText(frame, f"SEAT AREA ({self.total_seats} seats)", (x1r, y1r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        self._draw_status(frame, occupied)
        return frame, occupied

    def _draw_status(self, frame, occupied):
        h, w = frame.shape[:2]
        color = (0, 255, 0) if occupied < self.total_seats else (0, 0, 255)
        cv2.rectangle(frame, (0, 0), (w, 36), (40, 40, 40), -1)
        cv2.putText(frame, f"SEATS: {self.total_seats}   OCCUPIED: {occupied}",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"FPS: {self.fps:.0f}", (w - 110, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # ----------------------------------------------------------------------- run
    def run(self, source):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")

        if self.roi is None and self.select_roi:
            ok, frame = cap.read()
            if ok:
                r = cv2.selectROI("Draw chair area, then press ENTER", frame,
                                  fromCenter=False, showCrosshair=True)
                if r[2] > 0 and r[3] > 0:
                    x, y, w, h = r
                    self.roi = (x, y, x + w, y + h)
                    self._roi_fixed = True
                cv2.destroyAllWindows()

        print(f"Counting: SEATS={self.total_seats or 'auto'} | chair area={self.roi} | ESC to exit")
        prev = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame, occupied = self.process(frame)
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(1e-6, time.time() - prev))
            prev = time.time()
            cv2.imshow("Classroom Occupancy - press ESC to exit", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()


def parse_roi(text):
    x1, y1, x2, y2 = [int(v) for v in text.split(",")]
    return x1, y1, x2, y2


def parse_args():
    parser = argparse.ArgumentParser(description="Classroom Occupancy live counter")
    parser.add_argument("--model", default="models/pretrained/yolo11n-pose.pt",
                        help="pose model (upper body + head). Non-pose models fall back to box head estimate")
    parser.add_argument("--chair-model", default="models/pretrained/yolo11n.pt",
                        help="detect model used to find the chairs (COCO class 56)")
    parser.add_argument("--source", default=0, type=int,
                        help="camera index (0) or video file path")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="person/head detection confidence threshold")
    parser.add_argument("--chair-conf", type=float, default=0.25,
                        help="chair detection confidence threshold")
    parser.add_argument("--seats", type=int, default=0,
                        help="total seats (0 = auto from chair detections)")
    parser.add_argument("--roi", default=None,
                        help="manual chair area box: x1,y1,x2,y2 (skips auto)")
    parser.add_argument("--select-roi", action="store_true",
                        help="draw the chair area box by hand on the first frame")
    parser.add_argument("--imgsz", type=int, default=480,
                        help="inference size for people (lower = faster; 320/480/640)")
    parser.add_argument("--chair-imgsz", type=int, default=320,
                        help="inference size for chairs (big objects, keep low)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    occ = ClassroomOccupancy(
        args.model, args.conf, args.seats,
        parse_roi(args.roi) if args.roi else None,
        args.chair_model, args.chair_conf, select_roi=args.select_roi,
        imgsz=args.imgsz, chair_imgsz=args.chair_imgsz,
    )
    occ.run(args.source)
