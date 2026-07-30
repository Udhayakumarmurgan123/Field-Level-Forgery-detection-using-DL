"""
Draws field bounding boxes + tampering/forgery-type labels onto a sample of dataset
images for manual visual verification. Genuine fields are outlined green, tampered
fields red with the forgery type annotated.

Usage:
    python scripts/visual_inspect_dataset.py --n 6
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.schema import DocumentAnnotation  # noqa: E402

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def draw_annotation(doc: DocumentAnnotation, dataset_root: Path) -> Image.Image:
    img = Image.open(dataset_root / doc.image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, 13)

    for f in doc.fields:
        color = (220, 30, 30) if f.is_tampered else (30, 160, 60)
        b = f.bbox
        draw.rectangle([b.x_min, b.y_min, b.x_max, b.y_max], outline=color, width=3)
        label = f.field_name if not f.is_tampered else f"{f.field_name} [{f.forgery_type}]"
        text_y = max(0, b.y_min - 16)
        draw.rectangle([b.x_min, text_y, b.x_min + 8 * len(label), text_y + 14], fill=color)
        draw.text((b.x_min + 2, text_y), label, font=font, fill=(255, 255, 255))

    status = "FORGED" if doc.is_forged else "GENUINE"
    draw.text((10, img.height - 20), f"{doc.document_id} | {status}", font=font, fill=(0, 0, 0))
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6, help="number of documents to render")
    parser.add_argument("--out", type=str, default="outputs/audit/visual_inspection")
    args = parser.parse_args()

    ann_path = ROOT / "data" / "raw" / "annotations.json"
    dataset_root = ROOT / "data" / "raw"
    data = json.load(open(ann_path))
    docs = [DocumentAnnotation.from_dict(d) for d in data["documents"]]

    rng = random.Random(0)
    sample = rng.sample(docs, min(args.n, len(docs)))

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for doc in sample:
        annotated = draw_annotation(doc, dataset_root)
        out_path = out_dir / f"{doc.document_id}_inspect.png"
        annotated.save(out_path)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
