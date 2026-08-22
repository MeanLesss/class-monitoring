"""
Train "class-monitor-ai" - transfer learning from the COCO-pretrained YOLO11n
onto the labeled classroom dataset (dataset/images + dataset/val/images,
single class: person).

Result is stored in the project as:
    models/tuned/class-monitor-ai.pt          <- used by app.py
    models/tuned/class-monitor-ai.meta.json   <- training metadata

Run (from this folder):
    .\\ai-venv\\Scripts\\python.exe train_class_monitor.py [--epochs 10] [--imgsz 640]
"""

import argparse
import json
import shutil
import time
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
BASE_WEIGHTS = ROOT / "models" / "pretrained" / "yolo11n.pt"
DATA_YAML = ROOT / "class_monitor.yaml"
OUT_DIR = ROOT / "models" / "tuned"


def main():
    ap = argparse.ArgumentParser(description="Fine-tune class-monitor-ai")
    ap.add_argument("--epochs", type=int, default=10, help="low epochs for a quick nice result")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="RTX 2060 6GB-friendly")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="class-monitor-ai")
    args = ap.parse_args()

    assert BASE_WEIGHTS.exists(), f"missing base weights: {BASE_WEIGHTS}"
    assert DATA_YAML.exists(), f"missing data yaml: {DATA_YAML}"

    model = YOLO(str(BASE_WEIGHTS))     # transfer learning from COCO weights
    t0 = time.time()
    results = model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=str(ROOT / "models" / "training"),
        name=args.name,
        exist_ok=True,
        patience=max(3, args.epochs // 2),   # early-stop if it plateaus
        verbose=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    assert best.exists(), "training finished but best.pt not found"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final = OUT_DIR / f"{args.name}.pt"
    shutil.copy2(best, final)

    meta = {
        "name": args.name,
        "base_weights": str(BASE_WEIGHTS.relative_to(ROOT)),
        "data": str(DATA_YAML.relative_to(ROOT)),
        "classes": model.names,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "trained_minutes": round((time.time() - t0) / 60, 1),
        "metrics": {k: (float(v) if hasattr(v, "item") else v)
                    for k, v in getattr(results, "results_dict", {}).items()},
        "run_dir": str(results.save_dir),
        "final_weights": str(final.relative_to(ROOT)),
    }
    meta_path = OUT_DIR / f"{args.name}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print("\n=== class-monitor-ai trained ===")
    print(f"weights : {final}")
    print(f"meta    : {meta_path}")
    print(f"mAP50-95: {meta['metrics'].get('metrics/mAP50-95(B)', 'n/a')}")
    print(f"mAP50   : {meta['metrics'].get('metrics/mAP50(B)', 'n/a')}")


if __name__ == "__main__":
    main()
