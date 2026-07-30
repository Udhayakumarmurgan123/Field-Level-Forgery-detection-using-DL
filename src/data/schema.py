"""
Annotation schema for the Field-Level Explainable Document Forgery Detection dataset.

Every synthetic (or real, later) document is described by a `DocumentAnnotation`.
Documents are organized into `document_group`s: a genuine document and every forged
variant derived from it share the same group id. This is what lets the dataset
validator prevent train/test leakage -- a group must live entirely in one split.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


FIELD_NAMES = ["name", "dob", "id_number", "address", "photo", "signature"]

FORGERY_TYPES = [
    "none",
    "text_replacement",
    "font_modification",
    "spacing_modification",
    "copy_paste",
    "image_region_replacement",
]


@dataclass
class BoundingBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def as_xywh_norm(self, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        """Convert to YOLO-format normalized (x_center, y_center, w, h)."""
        w = (self.x_max - self.x_min) / img_w
        h = (self.y_max - self.y_min) / img_h
        xc = (self.x_min + self.x_max) / 2 / img_w
        yc = (self.y_min + self.y_max) / 2 / img_h
        return xc, yc, w, h

    def valid(self, img_w: int, img_h: int) -> bool:
        return (
            0 <= self.x_min < self.x_max <= img_w
            and 0 <= self.y_min < self.y_max <= img_h
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class EvidenceAnnotation:
    """Ground-truth evidence describing *why* a field is considered tampered.
    For genuine fields this is an empty list.
    """
    evidence_type: str          # e.g. "font_mismatch", "noise_discontinuity", "ela_hotspot"
    description: str            # human-readable, used to sanity check / train explanation templates
    confidence: float = 1.0     # ground-truth confidence (synthetic data -> usually 1.0)


@dataclass
class FieldAnnotation:
    field_name: str                       # one of FIELD_NAMES
    bbox: BoundingBox
    is_tampered: bool
    forgery_type: str                     # one of FORGERY_TYPES ("none" if not tampered)
    evidence: list[EvidenceAnnotation] = field(default_factory=list)

    def __post_init__(self):
        if self.field_name not in FIELD_NAMES:
            raise ValueError(f"Unknown field_name '{self.field_name}'")
        if self.forgery_type not in FORGERY_TYPES:
            raise ValueError(f"Unknown forgery_type '{self.forgery_type}'")
        if self.is_tampered and self.forgery_type == "none":
            raise ValueError("Tampered field must have a forgery_type other than 'none'")
        if not self.is_tampered and self.forgery_type != "none":
            raise ValueError("Non-tampered field must have forgery_type == 'none'")


@dataclass
class DocumentAnnotation:
    document_id: str              # unique id for this specific image, e.g. "doc_0001_forged"
    document_group: str           # shared id across genuine + all forged variants, e.g. "doc_0001"
    image_path: str               # relative path to the image file
    image_width: int
    image_height: int
    is_forged: bool               # document-level label (True if ANY field is tampered)
    split: str                    # "train" | "val" | "test"
    fields: list[FieldAnnotation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def tampered_fields(self) -> list[FieldAnnotation]:
        return [f for f in self.fields if f.is_tampered]

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "document_group": self.document_group,
            "image_path": self.image_path,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "is_forged": self.is_forged,
            "split": self.split,
            "metadata": self.metadata,
            "fields": [
                {
                    "field_name": f.field_name,
                    "bbox": f.bbox.to_dict(),
                    "is_tampered": f.is_tampered,
                    "forgery_type": f.forgery_type,
                    "evidence": [asdict(e) for e in f.evidence],
                }
                for f in self.fields
            ],
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "DocumentAnnotation":
        fields_ = []
        for fd in d["fields"]:
            bbox = BoundingBox(**fd["bbox"])
            evidence = [EvidenceAnnotation(**e) for e in fd.get("evidence", [])]
            fields_.append(
                FieldAnnotation(
                    field_name=fd["field_name"],
                    bbox=bbox,
                    is_tampered=fd["is_tampered"],
                    forgery_type=fd["forgery_type"],
                    evidence=evidence,
                )
            )
        return DocumentAnnotation(
            document_id=d["document_id"],
            document_group=d["document_group"],
            image_path=d["image_path"],
            image_width=d["image_width"],
            image_height=d["image_height"],
            is_forged=d["is_forged"],
            split=d["split"],
            fields=fields_,
            metadata=d.get("metadata", {}),
        )
