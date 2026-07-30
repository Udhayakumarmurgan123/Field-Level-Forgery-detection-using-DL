"""
Inference-time wrapper around the trained YOLO field-localization model.

Falls back gracefully: if no trained weights are found yet, `is_available()`
returns False and the pipeline can fall back to a fixed heuristic layout (useful
so the rest of the system -- evidence extraction, classifier, explanation, UI --
can be developed/demoed even before YOLO training has produced weights).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FieldDetection:
    field_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x_min, y_min, x_max, y_max


class FieldLocalizer:
    def __init__(self, weights_path: Path, class_names: list[str], conf_threshold: float = 0.25):
        self.weights_path = Path(weights_path)
        self.class_names = class_names
        self.conf_threshold = conf_threshold
        self._model = None

    def is_available(self) -> bool:
        return self.weights_path.exists()

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(str(self.weights_path))
        return self._model

    def detect(self, image_bgr: np.ndarray) -> list[FieldDetection]:
        model = self._load()
        results = model.predict(image_bgr, conf=self.conf_threshold, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_idx = int(box.cls.item())
            conf = float(box.conf.item())
            x0, y0, x1, y1 = [int(v) for v in box.xyxy[0].tolist()]
            if cls_idx < len(self.class_names):
                detections.append(
                    FieldDetection(
                        field_name=self.class_names[cls_idx],
                        confidence=conf,
                        bbox=(x0, y0, x1, y1),
                    )
                )
        # keep only the highest-confidence detection per field class
        best_per_field: dict[str, FieldDetection] = {}
        for det in detections:
            existing = best_per_field.get(det.field_name)
            if existing is None or det.confidence > existing.confidence:
                best_per_field[det.field_name] = det
        return list(best_per_field.values())
