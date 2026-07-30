"""
Dataset validator.

Checks:
  1. Every image file referenced actually exists and its dimensions match the annotation.
  2. Every bounding box is valid (within image bounds, x_min<x_max, y_min<y_max).
  3. Field set per document matches the expected FIELD_NAMES exactly (no missing/extra fields).
  4. Tampered fields have non-empty, well-formed evidence; non-tampered fields have none.
  5. document-level `is_forged` flag is consistent with whether any field is tampered.
  6. No document_group appears in more than one split (train/val/test) -- leakage check.
  7. forgery_type values are drawn from the allowed vocabulary.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from src.data.schema import DocumentAnnotation, FIELD_NAMES, FORGERY_TYPES


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = [f"Errors: {len(self.errors)} | Warnings: {len(self.warnings)}"]
        lines += [f"  [ERROR] {e}" for e in self.errors]
        lines += [f"  [WARN]  {w}" for w in self.warnings]
        return "\n".join(lines)


def validate_dataset(annotations_json: Path, dataset_root: Path) -> ValidationReport:
    report = ValidationReport()
    data = json.load(open(annotations_json))
    docs = [DocumentAnnotation.from_dict(d) for d in data["documents"]]

    group_splits: dict[str, set[str]] = {}
    seen_ids = set()

    for doc in docs:
        # duplicate id check
        if doc.document_id in seen_ids:
            report.errors.append(f"Duplicate document_id: {doc.document_id}")
        seen_ids.add(doc.document_id)

        # image existence + dimension check
        img_path = dataset_root / doc.image_path
        if not img_path.exists():
            report.errors.append(f"{doc.document_id}: image file not found at {img_path}")
            continue
        with Image.open(img_path) as im:
            if im.size != (doc.image_width, doc.image_height):
                report.errors.append(
                    f"{doc.document_id}: annotation size {(doc.image_width, doc.image_height)} "
                    f"!= actual image size {im.size}"
                )

        # field completeness
        present_fields = {f.field_name for f in doc.fields}
        missing = set(FIELD_NAMES) - present_fields
        extra = present_fields - set(FIELD_NAMES)
        if missing:
            report.errors.append(f"{doc.document_id}: missing fields {sorted(missing)}")
        if extra:
            report.errors.append(f"{doc.document_id}: unexpected fields {sorted(extra)}")

        # per-field checks
        any_tampered = False
        for f in doc.fields:
            if not f.bbox.valid(doc.image_width, doc.image_height):
                report.errors.append(
                    f"{doc.document_id}/{f.field_name}: invalid bbox {f.bbox} for image "
                    f"size {(doc.image_width, doc.image_height)}"
                )
            if f.forgery_type not in FORGERY_TYPES:
                report.errors.append(f"{doc.document_id}/{f.field_name}: unknown forgery_type '{f.forgery_type}'")
            if f.is_tampered:
                any_tampered = True
                if not f.evidence:
                    report.errors.append(f"{doc.document_id}/{f.field_name}: tampered field has no evidence")
                for ev in f.evidence:
                    if not ev.description or len(ev.description.strip()) < 5:
                        report.errors.append(
                            f"{doc.document_id}/{f.field_name}: evidence description too short/empty"
                        )
            else:
                if f.evidence:
                    report.warnings.append(
                        f"{doc.document_id}/{f.field_name}: non-tampered field has evidence entries (ignored downstream)"
                    )

        # document-level consistency
        if any_tampered != doc.is_forged:
            report.errors.append(
                f"{doc.document_id}: document.is_forged={doc.is_forged} but "
                f"field-level tampering={'present' if any_tampered else 'absent'}"
            )

        # split bookkeeping
        group_splits.setdefault(doc.document_group, set()).add(doc.split)

    # leakage check
    for gid, splits in group_splits.items():
        if len(splits) > 1:
            report.errors.append(f"LEAKAGE: document_group '{gid}' appears in multiple splits: {splits}")

    if not docs:
        report.errors.append("Dataset contains zero documents.")

    return report


if __name__ == "__main__":
    import sys
    root = Path(__file__).resolve().parents[2]
    ann_path = root / "data" / "raw" / "annotations.json"
    rep = validate_dataset(ann_path, root / "data" / "raw")
    print(rep.summary())
    sys.exit(0 if rep.is_valid else 1)
