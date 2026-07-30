"""
Field-level forgery classifier.

Rather than a heavy CNN/ViT (impractical to train well on CPU with a small
synthetic dataset), this stage trains a lightweight gradient-boosted / logistic
model on top of the classical evidence-signal *scores* themselves (ELA, noise
discontinuity, edge seam, spacing irregularity). This keeps the whole pipeline
CPU-fast, keeps every input to the classifier human-interpretable (each feature
IS an evidence signal that gets shown to the user), and gives a natural upgrade
path: swap in a CNN/ViT backbone later and concatenate its embedding with these
same evidence features without changing anything downstream.

Feature vector per field (fixed order, missing signals default to 0.0):
    [ela_hotspot, noise_discontinuity, edge_seam_discontinuity, spacing_irregularity]
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.evidence_extraction.classical_evidence import EvidenceSignal

FEATURE_ORDER = [
    "ela_hotspot",
    "noise_discontinuity",
    "edge_seam_discontinuity",
    "spacing_irregularity",
]


def signals_to_feature_vector(signals: list[EvidenceSignal]) -> np.ndarray:
    by_type = {s.evidence_type: s.score for s in signals}
    return np.array([by_type.get(name, 0.0) for name in FEATURE_ORDER], dtype=np.float32)


@dataclass
class FieldForgeryClassifier:
    model: GradientBoostingClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model = GradientBoostingClassifier(
            n_estimators=150, max_depth=2, learning_rate=0.08, random_state=42
        )
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Classifier not fitted/loaded.")
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: Path) -> None:
        joblib.dump(self.model, path)

    @staticmethod
    def load(path: Path) -> "FieldForgeryClassifier":
        model = joblib.load(path)
        return FieldForgeryClassifier(model=model)

    def feature_importance(self) -> dict[str, float]:
        if self.model is None:
            return {}
        return dict(zip(FEATURE_ORDER, self.model.feature_importances_.tolist()))
