"""
End-to-end inference pipeline:

    image -> [YOLO field localization] -> per-field crops
          -> [classical evidence extraction] -> evidence signals
          -> [lightweight forgery classifier] -> tampering probability per field
          -> [template explanation generator] -> grounded NL explanation
          -> annotated image + structured result dict

If YOLO weights aren't trained yet, falls back to the same fixed heuristic
layout used by the synthetic generator (keeps the UI usable at every dev stage).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from src.data.schema import FIELD_NAMES
from src.evidence_extraction.classical_evidence import extract_field_evidence
from src.explanation.template_explainer import generate_document_summary, generate_field_explanation
from src.forgery_detection.field_classifier import FieldForgeryClassifier, signals_to_feature_vector
from src.localization.field_localizer import FieldLocalizer, FieldDetection
from src.utils.fonts import load_font

ROOT = Path(__file__).resolve().parents[2]

FIELD_KIND = {
    "name": "text", "dob": "text", "id_number": "text", "address": "text",
    "photo": "photo", "signature": "signature",
}

# fixed fallback layout mirroring src/data/synthetic_generator.py, used only if
# YOLO weights are not available yet
def _fallback_layout(img_w: int, img_h: int) -> list[FieldDetection]:
    margin = 40
    photo_w, photo_h = 150, 180
    layout = [
        ("photo", margin, margin + 20, photo_w, photo_h),
        ("name", margin + photo_w + 40, margin + 30, 420, 34),
        ("dob", margin + photo_w + 40, margin + 90, 260, 30),
        ("id_number", margin + photo_w + 40, margin + 145, 260, 30),
        ("address", margin + photo_w + 40, margin + 200, 480, 30),
        ("signature", img_w - 260, img_h - 90, 200, 60),
    ]
    return [
        FieldDetection(name, 1.0, (x, y, x + w, y + h))
        for name, x, y, w, h in layout
    ]


class PrototypeDocumentAnalysisPipeline:
    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path or (ROOT / "configs" / "config.yaml")
        self.cfg = yaml.safe_load(open(self.config_path))

        yolo_weights = ROOT / self.cfg["paths"]["checkpoints"] / "field_localization_best.pt"
        self.localizer = FieldLocalizer(yolo_weights, FIELD_NAMES)

        clf_path = ROOT / self.cfg["forgery_classifier"]["model_path"]
        self.classifier: FieldForgeryClassifier | None = None
        if clf_path.exists():
            self.classifier = FieldForgeryClassifier.load(clf_path)

        self.ev_cfg = self.cfg["evidence"]
        self.outputs_dir = ROOT / self.cfg["paths"]["annotated"]
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def _get_detections(self, image_bgr: np.ndarray) -> tuple[list[FieldDetection], str]:
        if self.localizer.is_available():
            dets = self.localizer.detect(image_bgr)
            if dets:
                return dets, "yolo"
        h, w = image_bgr.shape[:2]
        return _fallback_layout(w, h), "fallback_heuristic_layout"

    def analyze_image(self, image_path: Path) -> dict[str, Any]:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise ValueError(f"Could not read image at {image_path}")
        h, w = image_bgr.shape[:2]

        detections, mode = self._get_detections(image_bgr)

        field_results = []
        tampered_field_name = None
        max_tampered_conf = 0.0

        for det in detections:
            kind = FIELD_KIND.get(det.field_name, "text")
            signals = extract_field_evidence(
                image_bgr, det.bbox, kind,
                ela_quality=self.ev_cfg["ela_quality"],
                edge_low=self.ev_cfg["edge_threshold_low"],
                edge_high=self.ev_cfg["edge_threshold_high"],
            )
            feats = signals_to_feature_vector(signals).reshape(1, -1)

            if self.classifier is not None:
                tamper_conf = float(self.classifier.predict_proba(feats)[0])
            else:
                # fallback: simple max-of-signals heuristic if classifier isn't trained yet
                tamper_conf = float(max((s.score for s in signals), default=0.0))

            is_tampered = tamper_conf >= 0.5
            field_exp = generate_field_explanation(det.field_name, is_tampered, tamper_conf, signals)

            if is_tampered and tamper_conf > max_tampered_conf:
                max_tampered_conf = tamper_conf
                tampered_field_name = det.field_name

            field_results.append({
                "field_name": det.field_name,
                "tampering_status": "tampered" if is_tampered else "genuine",
                "forgery_type": field_exp.likely_forgery_type,
                "confidence": tamper_conf,
                "bbox": det.bbox,
                "evidence": [
                    {"type": s.evidence_type, "confidence": s.score, "description": s.description}
                    for s in signals
                ],
                "explanation": field_exp.explanation,
            })

        overall_forged = tampered_field_name is not None
        overall_conf = max_tampered_conf if overall_forged else (
            1.0 - max((r["confidence"] for r in field_results), default=0.0)
        )

        field_explanations = [
            generate_field_explanation(r["field_name"], r["tampering_status"] == "tampered", r["confidence"],
                                        [])  # summary doesn't need raw signals again
            for r in field_results
        ]
        summary_text = generate_document_summary(field_explanations)

        annotated_path = self._draw_annotated(image_path, field_results)

        note = (
            "Field localization via trained YOLO model."
            if mode == "yolo" else
            "YOLO weights not found -- using fixed heuristic layout as a fallback. "
            "Run scripts/train_yolo.py to enable learned localization."
        )
        if self.classifier is None:
            note += " Forgery classifier not found -- using raw evidence-signal heuristic as fallback."

        return {
            "overall_status": "FORGED" if overall_forged else "GENUINE",
            "overall_confidence": overall_conf,
            "mode": mode,
            "note": note,
            "tampered_field": tampered_field_name,
            "field_results": field_results,
            "annotated_image_path": str(annotated_path),
            "summary": summary_text,
        }

    def _draw_annotated(self, image_path: Path, field_results: list[dict[str, Any]]) -> Path:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        font = load_font("bold", 13)

        for r in field_results:
            x0, y0, x1, y1 = r["bbox"]
            tampered = r["tampering_status"] == "tampered"
            color = (220, 30, 30) if tampered else (30, 160, 60)
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
            label = f"{r['field_name']} ({r['confidence']:.2f})"
            text_y = max(0, y0 - 16)
            draw.rectangle([x0, text_y, x0 + 8 * len(label), text_y + 14], fill=color)
            draw.text((x0 + 2, text_y), label, font=font, fill=(255, 255, 255))

        out_path = self.outputs_dir / f"{Path(image_path).stem}_annotated.png"
        img.save(out_path)
        return out_path