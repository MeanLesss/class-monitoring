"""
Classroom Occupancy - dataset collector.

Captures photos of classroom seats (occupied vs empty) from the webcam so you
can test YOLO person count + pose landmarks later in verify_yolo.ipynb.

Controls:
    SPACE  -> save the current frame
    R      -> rotate the room's angle/position (just a hint to move the camera)
    ESC    -> exit

Usage:
    python collect_dataset.py --count 150
    python collect_dataset.py --out dataset/images --label occupied
"""

import argparse
import os
import time

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Collect classroom photos")
    parser.add_argument("--out", default="dataset/images",
                        help="output folder for captured photos")
    parser.add_argument("--count", type=int, default=150,
                        help="target number of photos (100-200 recommended)")
    parser.add_argument("--prefix", default="seat",
                        help="file name prefix")
    parser.add_argument("--camera", type=int, default=0,
                        help="camera index")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    existing = len([f for f in os.listdir(args.out) if f.endswith(".jpg")])
    target = existing + args.count
    saved = existing

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    print(f"Saving photos to: {os.path.abspath(args.out)}")
    print("SPACE=save | R=hint to change angle | ESC=exit")
    print(f"Already saved: {saved}/{target}")

    while saved < target:
        ok, frame = cap.read()
        if not ok:
            break
        hint = ""
        if saved > existing and saved % 10 == 0 and saved % 25 != 0:
            hint = "  <-- try a new camera angle or occupied/empty setup"
        cv2.putText(frame, f"Saved: {saved}/{target}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE=save  R=angle hint  ESC=exit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if hint:
            cv2.putText(frame, hint, (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.imshow("Collect classroom photos", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key in (32, 13):  # SPACE / ENTER
            name = os.path.join(args.out, f"{args.prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{saved}.jpg")
            cv2.imwrite(name, frame)
            saved += 1
            print(f"[{saved}/{target}] saved {name}")
        elif key == ord("r"):
            print("Hint: move the camera to a different classroom angle "
                  "(high/eye-level/side) and press SPACE")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. {saved - existing} new photos in {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
