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


@dataclass(frozen=True)
class FieldDetection:
    field_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x_min, y_min, x_max, y_max


@dataclass(frozen=True)
class FieldLocalizerConfig:
    """
    Configuration object grouping all parameters required to initialise
    :class:`FieldLocalizer`.  Using a dedicated config eliminates the long
    parameter list in ``FieldLocalizer.__init__`` and makes it easier to pass
    around a single coherent object.
    """
    weights_path: Path
    class_names: list[str]
    conf_threshold: float = 0.25


class FieldLocalizer:
    """
    Wrapper around a YOLO model that detects fields in an image.
    """

    def __init__(self, config: FieldLocalizerConfig):
        # Store configuration values directly; they are immutable thanks to the
        # frozen dataclass, which prevents accidental modification.
        self.weights_path: Path = Path(config.weights_path)
        self.class_names: list[str] = config.class_names
        self.conf_threshold: float = config.conf_threshold
        self._model = None

    def is_available(self) -> bool:
        """Return ``True`` if the model weights exist on disk."""
        return self.weights_path.exists()

    def _load(self):
        """Lazy‑load the YOLO model; executed only once."""
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(str(self.weights_path))
        return self._model

    def _best_detection_per_field(self, detections: list[FieldDetection]) -> list[FieldDetection]:
        """
        Keep only the highest‑confidence detection for each field class.
        """
        best_per_field: dict[str, FieldDetection] = {}
        for det in detections:
            existing = best_per_field.get(det.field_name)
            if existing is None or det.confidence > existing.confidence:
                best_per_field[det.field_name] = det
        return list(best_per_field.values())

    def detect(self, image_bgr: np.ndarray) -> list[FieldDetection]:
        """Run the model on ``image_bgr`` and return the best detection per field."""
        model = self._load()
        results = model.predict(image_bgr, conf=self.conf_threshold, verbose=False)[0]
        detections: list[FieldDetection] = []

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

        return self._best_detection_per_field(detections)