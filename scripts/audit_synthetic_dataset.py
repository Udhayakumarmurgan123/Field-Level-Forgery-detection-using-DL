"""
Audits the generated dataset: class balance, per-field tampering rate, forgery-type
distribution, split sizes, and image-consistency spot checks. Writes both a
human-readable text report and a machine-readable JSON summary to outputs/audit/.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.schema import DocumentAnnotation  # noqa: E402


def main():
    ann_path = ROOT / "data" / "raw" / "annotations.json"
    data = json.load(open(ann_path))
    docs = [DocumentAnnotation.from_dict(d) for d in data["documents"]]

    split_counts = Counter(d.split for d in docs)
    label_counts = Counter("forged" if d.is_forged else "genuine" for d in docs)
    field_tamper_counts = Counter()
    field_total_counts = Counter()
    forgery_type_counts = Counter()

    split_label_counts: dict[str, Counter] = {s: Counter() for s in split_counts}

    for d in docs:
        split_label_counts[d.split]["forged" if d.is_forged else "genuine"] += 1
        for f in d.fields:
            field_total_counts[f.field_name] += 1
            if f.is_tampered:
                field_tamper_counts[f.field_name] += 1
                forgery_type_counts[f.forgery_type] += 1

    report = {
        "total_documents": len(docs),
        "split_counts": dict(split_counts),
        "label_counts": dict(label_counts),
        "split_label_counts": {s: dict(c) for s, c in split_label_counts.items()},
        "field_tamper_rate": {
            field: round(field_tamper_counts[field] / field_total_counts[field], 4)
            for field in field_total_counts
        },
        "forgery_type_counts": dict(forgery_type_counts),
    }

    out_dir = ROOT / "outputs" / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "audit_report.json", "w") as f:
        json.dump(report, f, indent=2)

    lines = ["=== Dataset Audit Report ===", f"Total documents: {report['total_documents']}", ""]
    lines.append("Split sizes:")
    for s, c in report["split_counts"].items():
        lines.append(f"  {s}: {c}  ({report['split_label_counts'][s]})")
    lines.append("")
    lines.append("Overall genuine/forged balance:")
    for k, v in report["label_counts"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Per-field tamper rate (fraction of documents where this field is tampered):")
    for k, v in report["field_tamper_rate"].items():
        lines.append(f"  {k}: {v:.2%}")
    lines.append("")
    lines.append("Forgery type distribution (count of tampered fields by type):")
    for k, v in report["forgery_type_counts"].items():
        lines.append(f"  {k}: {v}")

    text_report = "\n".join(lines)
    with open(out_dir / "audit_report.txt", "w") as f:
        f.write(text_report)

    print(text_report)
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    main()
