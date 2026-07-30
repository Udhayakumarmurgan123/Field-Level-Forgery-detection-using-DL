"""
Generate the synthetic dataset: N document groups, each containing one genuine and
one forged document, with group-level train/val/test split assignment (so a
genuine+forged pair from the same underlying document never crosses splits).

Usage:
    python scripts/generate_synthetic_dataset.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.synthetic_generator import generate_document_group  # noqa: E402


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    synth_cfg = cfg["synthetic"]
    out_root = ROOT / cfg["paths"]["dataset_root"]
    out_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(synth_cfg["seed"])
    n = synth_cfg["num_documents"]
    img_w, img_h = synth_cfg["image_width"], synth_cfg["image_height"]

    group_ids = [f"doc_{i:04d}" for i in range(n)]
    rng.shuffle(group_ids)
    n_train = int(len(group_ids) * synth_cfg["train_split"])
    n_val = max(1, int(len(group_ids) * (1 - synth_cfg["train_split"]) / 2))
    train_groups = set(group_ids[:n_train])
    val_groups = set(group_ids[n_train:n_train + n_val])
    test_groups = set(group_ids[n_train + n_val:])

    def split_of(gid: str) -> str:
        if gid in train_groups:
            return "train"
        if gid in val_groups:
            return "val"
        return "test"

    all_annotations = []
    images_dir = out_root / "images"
    for gid in group_ids:
        anns = generate_document_group(gid, img_w, img_h, images_dir, rng)
        split = split_of(gid)
        for a in anns:
            a.split = split
            a.image_path = f"images/{a.image_path}"
        all_annotations.extend(anns)

    dataset = {
        "fields": cfg["fields"],
        "forgery_types": cfg["forgery_types"],
        "documents": [a.to_dict() for a in all_annotations],
    }
    out_json = out_root / "annotations.json"
    with open(out_json, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Generated {len(all_annotations)} documents from {n} groups.")
    print(f"  train groups: {len(train_groups)} | val groups: {len(val_groups)} | test groups: {len(test_groups)}")
    print(f"Annotations written to: {out_json}")
    print(f"Images written to: {images_dir}")


if __name__ == "__main__":
    main()
