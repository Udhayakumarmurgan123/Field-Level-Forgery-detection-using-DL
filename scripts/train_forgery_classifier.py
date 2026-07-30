"""
Builds a (features, label) table by running classical evidence extraction over
every field crop in the dataset (using ground-truth bboxes -- this stage trains
independently of YOLO's localization accuracy), then trains the lightweight
GradientBoosting forgery classifier and reports held-out metrics.

Usage:
    python scripts/train_forgery_classifier.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from sklearn.metrics import classification_report, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.schema import DocumentAnnotation  # noqa: E402
from src.evidence_extraction.classical_evidence import extract_field_evidence  # noqa: E402
from src.forgery_detection.field_classifier import (  # noqa: E402
    FieldForgeryClassifier,
    signals_to_feature_vector,
)

FIELD_KIND = {
    "name": "text", "dob": "text", "id_number": "text", "address": "text",
    "photo": "photo", "signature": "signature",
}


def build_feature_table(
    docs: list[DocumentAnnotation],
    dataset_root: Path,
    ela_quality: int,
    edge_low: int,
    edge_high: int,
    rng: np.random.Generator,
    jitter_px: int = 4,
    jitter_repeats: int = 3,
):
    """Builds features from GROUND-TRUTH boxes, but perturbs each box by up to
    `jitter_px` pixels on every side (repeated `jitter_repeats` times for the
    train split) to simulate the small localization imprecision that the
    trained YOLO model actually produces at inference time. Without this, the
    classifier overfits to pixel-perfect boxes and becomes unstable on real
    YOLO output (verified empirically: a 2px shift flipped a prediction before
    this fix). Val/test splits keep one exact-box sample each, matching how
    we report held-out metrics.
    """
    X, y, splits, meta = [], [], [], []
    image_cache: dict[str, np.ndarray] = {}

    for doc in docs:
        if doc.image_path not in image_cache:
            img = cv2.imread(str(dataset_root / doc.image_path))
            image_cache[doc.image_path] = img
        img = image_cache[doc.image_path]
        if img is None:
            continue
        h, w = img.shape[:2]

        for f in doc.fields:
            base = (f.bbox.x_min, f.bbox.y_min, f.bbox.x_max, f.bbox.y_max)
            n_variants = jitter_repeats if doc.split == "train" else 1
            for i in range(n_variants):
                if i == 0:
                    bbox = base  # always include the exact box once
                else:
                    dx0, dy0, dx1, dy1 = rng.integers(-jitter_px, jitter_px + 1, size=4)
                    bbox = (
                        max(0, base[0] + dx0), max(0, base[1] + dy0),
                        min(w, base[2] + dx1), min(h, base[3] + dy1),
                    )
                    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                        bbox = base
                signals = extract_field_evidence(
                    img, bbox, FIELD_KIND[f.field_name],
                    ela_quality=ela_quality, edge_low=edge_low, edge_high=edge_high,
                )
                X.append(signals_to_feature_vector(signals))
                y.append(1 if f.is_tampered else 0)
                splits.append(doc.split)
                meta.append({"document_id": doc.document_id, "field_name": f.field_name, "forgery_type": f.forgery_type})

    return np.stack(X), np.array(y), np.array(splits), meta


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    ev_cfg = cfg["evidence"]
    dataset_root = ROOT / cfg["paths"]["dataset_root"]
    ann_path = dataset_root / "annotations.json"
    data = json.load(open(ann_path))
    docs = [DocumentAnnotation.from_dict(d) for d in data["documents"]]

    print("Extracting classical evidence features for all fields "
          "(ground-truth bboxes, with jitter augmentation on train split to match real YOLO imprecision)...")
    rng = np.random.default_rng(42)
    X, y, splits, meta = build_feature_table(
        docs, dataset_root, ev_cfg["ela_quality"], ev_cfg["edge_threshold_low"], ev_cfg["edge_threshold_high"],
        rng=rng, jitter_px=4, jitter_repeats=3,
    )
    print(f"Feature table: X={X.shape}, positive rate={y.mean():.3f}")

    train_mask = splits == "train"
    test_mask = splits == "test"
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    clf = FieldForgeryClassifier()
    clf.fit(X_train, y_train)

    proba_test = clf.predict_proba(X_test)
    preds_test = (proba_test >= 0.5).astype(int)

    print("\n=== Held-out TEST split report ===")
    print(classification_report(y_test, preds_test, target_names=["genuine", "tampered"], zero_division=0))
    try:
        auc = roc_auc_score(y_test, proba_test)
        print(f"ROC-AUC: {auc:.4f}")
    except ValueError:
        print("ROC-AUC: undefined (only one class present in test split)")

    print("\nFeature importances:")
    for k, v in clf.feature_importance().items():
        print(f"  {k}: {v:.4f}")

    ckpt_dir = ROOT / cfg["paths"]["checkpoints"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_path = ROOT / cfg["forgery_classifier"]["model_path"]
    clf.save(out_path)
    print(f"\nSaved classifier to {out_path}")


if __name__ == "__main__":
    main()
