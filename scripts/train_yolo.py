"""
Trains a YOLOv8 model to localize the six document fields (name, dob, id_number,
address, photo, signature). Defaults are tuned to be CPU-friendly (small model,
small image size, small batch) -- adjust in configs/config.yaml if you have a GPU.

Usage:
    python scripts/train_yolo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    yolo_cfg = cfg["yolo"]
    data_yaml = ROOT / cfg["paths"]["yolo_dataset_root"] / "data.yaml"
    ckpt_dir = ROOT / cfg["paths"]["checkpoints"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(yolo_cfg["model"])  # e.g. yolov8n.pt (downloads pretrained COCO weights, fine-tuned here)
    results = model.train(
        data=str(data_yaml),
        epochs=yolo_cfg["epochs"],
        imgsz=yolo_cfg["imgsz"],
        batch=yolo_cfg["batch"],
        device=yolo_cfg["device"],
        project=str(ckpt_dir),
        name="field_localization",
        exist_ok=True,
        verbose=True,
        patience=10,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    final_dest = ckpt_dir / "field_localization_best.pt"
    if best_weights.exists():
        import shutil
        shutil.copy(best_weights, final_dest)
        print(f"Best weights copied to {final_dest}")
    else:
        print(f"WARNING: expected best weights at {best_weights} not found")


if __name__ == "__main__":
    main()
