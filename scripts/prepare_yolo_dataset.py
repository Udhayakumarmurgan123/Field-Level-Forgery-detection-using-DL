"""
Converts the JSON dataset (data/raw/annotations.json) into YOLO detection format:

    data/yolo/
      images/{train,val,test}/*.png   (copied from data/raw/images)
      labels/{train,val,test}/*.txt   (class x_center y_center w h, normalized)
      data.yaml

The split assignment already present in each document's `split` field is reused
as-is (it was assigned at the document_group level in generation), so there is
still no leakage after conversion.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.schema import DocumentAnnotation, FIELD_NAMES  # noqa: E402

CLASS_TO_IDX = {name: i for i, name in enumerate(FIELD_NAMES)}


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    raw_root = ROOT / cfg["paths"]["dataset_root"]
    yolo_root = ROOT / cfg["paths"]["yolo_dataset_root"]

    ann_path = raw_root / "annotations.json"
    data = json.load(open(ann_path))
    docs = [DocumentAnnotation.from_dict(d) for d in data["documents"]]

    for split in ["train", "val", "test"]:
        (yolo_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0}
    for doc in docs:
        split = doc.split
        src_img = raw_root / doc.image_path
        dst_img = yolo_root / "images" / split / f"{doc.document_id}.png"
        shutil.copy(src_img, dst_img)

        label_lines = []
        for f in doc.fields:
            xc, yc, w, h = f.bbox.as_xywh_norm(doc.image_width, doc.image_height)
            cls_idx = CLASS_TO_IDX[f.field_name]
            label_lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        dst_label = yolo_root / "labels" / split / f"{doc.document_id}.txt"
        dst_label.write_text("\n".join(label_lines) + "\n")
        counts[split] += 1

    data_yaml = {
        "path": str(yolo_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(FIELD_NAMES)},
    }
    with open(yolo_root / "data.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f)

    print(f"YOLO dataset written to {yolo_root}")
    print(f"Counts: {counts}")
    print(f"data.yaml: {yolo_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
