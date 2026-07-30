"""
Synthetic document generator.

Produces ID-card-style document images (name / dob / id_number / address / photo /
signature) as PNGs, plus a genuine document and one-or-more forged variants per
document group, each with exact ground-truth field bounding boxes, tampering
labels, forgery types, and evidence annotations.

Because everything is generated, we know EXACTLY where each field is and EXACTLY
what was changed -- this is what makes automatic evidence-annotation possible
without any human labeling.
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.data.schema import (
    BoundingBox,
    DocumentAnnotation,
    EvidenceAnnotation,
    FieldAnnotation,
    FORGERY_TYPES,
)

FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_OBLIQUE = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

FIRST_NAMES = ["JOHN", "MARY", "ROBERT", "PRIYA", "AHMED", "LINH", "CARLOS", "ELENA", "KWAME", "YUKI"]
LAST_NAMES = ["SMITH", "GARCIA", "KUMAR", "NGUYEN", "MULLER", "OKAFOR", "SILVA", "TANAKA", "IVANOV", "PEREZ"]
STREETS = ["MAPLE AVE", "5TH STREET", "PARK ROAD", "RIVERSIDE DR", "CHURCH LANE", "MARKET ST"]
CITIES = ["SPRINGFIELD", "RIVERTON", "FAIRVIEW", "GREENVILLE", "ASHFORD", "MILTON"]


def _rand_name() -> str:
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def _rand_dob() -> str:
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.randint(1955, 2003)
    return f"{day:02d}/{month:02d}/{year}"


def _rand_id() -> str:
    return "".join(random.choices(string.digits, k=9))


def _rand_address() -> str:
    num = random.randint(1, 999)
    return f"{num} {random.choice(STREETS)}, {random.choice(CITIES)}"


@dataclass
class FieldLayout:
    name: str
    x: int
    y: int
    w: int
    h: int
    kind: str  # "text" | "photo" | "signature"


def _layout(img_w: int, img_h: int) -> list[FieldLayout]:
    """Fixed layout resembling a simple ID card. All positions are deterministic
    given the canvas size, which keeps bounding boxes exact."""
    margin = 40
    photo_w, photo_h = 150, 180
    return [
        FieldLayout("photo", margin, margin + 20, photo_w, photo_h, "photo"),
        FieldLayout("name", margin + photo_w + 40, margin + 30, 420, 34, "text"),
        FieldLayout("dob", margin + photo_w + 40, margin + 90, 260, 30, "text"),
        FieldLayout("id_number", margin + photo_w + 40, margin + 145, 260, 30, "text"),
        FieldLayout("address", margin + photo_w + 40, margin + 200, 480, 30, "text"),
        FieldLayout("signature", img_w - 260, img_h - 90, 200, 60, "signature"),
    ]


def _draw_photo(draw: ImageDraw.ImageDraw, box: FieldLayout, seed_color: tuple[int, int, int]):
    x0, y0, x1, y1 = box.x, box.y, box.x + box.w, box.y + box.h
    draw.rectangle([x0, y0, x1, y1], fill=(220, 220, 225), outline=(90, 90, 90), width=2)
    # crude "face" placeholder so the region isn't flat (helps evidence extraction be meaningful)
    cx, cy = x0 + box.w // 2, y0 + box.h // 2 - 10
    draw.ellipse([cx - 35, cy - 45, cx + 35, cy + 35], fill=seed_color)
    draw.ellipse([cx - 55, cy + 30, cx + 55, cy + 90], fill=seed_color)


def _draw_signature(draw: ImageDraw.ImageDraw, box: FieldLayout, rng: random.Random, color=(20, 20, 60)):
    x0, y0 = box.x, box.y + box.h // 2
    points = [(x0, y0)]
    x = x0
    for _ in range(10):
        x += rng.randint(15, 25)
        y = y0 + rng.randint(-box.h // 2 + 5, box.h // 2 - 5)
        points.append((x, y))
    draw.line(points, fill=color, width=2, joint="curve")


def render_genuine_document(img_w: int, img_h: int, rng: random.Random) -> tuple[Image.Image, dict[str, str], list[FieldLayout]]:
    img = Image.new("RGB", (img_w, img_h), color=(250, 248, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img_w - 1, img_h - 1], outline=(30, 60, 120), width=6)
    title_font = ImageFont.truetype(FONT_BOLD, 26)
    draw.text((img_w // 2 - 140, 10), "NATIONAL IDENTITY CARD", font=title_font, fill=(30, 60, 120))

    layouts = _layout(img_w, img_h)
    values = {
        "name": _rand_name(),
        "dob": _rand_dob(),
        "id_number": _rand_id(),
        "address": _rand_address(),
    }
    label_font = ImageFont.truetype(FONT_REGULAR, 12)
    value_font = ImageFont.truetype(FONT_REGULAR, 20)
    seed_color = (rng.randint(150, 220), rng.randint(120, 190), rng.randint(110, 180))

    for fl in layouts:
        if fl.kind == "photo":
            _draw_photo(draw, fl, seed_color)
        elif fl.kind == "signature":
            draw.rectangle([fl.x, fl.y, fl.x + fl.w, fl.y + fl.h], outline=(150, 150, 150), width=1)
            _draw_signature(draw, fl, rng)
        else:
            draw.text((fl.x, fl.y - 14), fl.name.replace("_", " ").upper(), font=label_font, fill=(120, 120, 120))
            draw.text((fl.x, fl.y), values[fl.name], font=value_font, fill=(10, 10, 10))

    return img, values, layouts


def _tamper_text_field(
    img: Image.Image,
    fl: FieldLayout,
    forgery_type: str,
    rng: random.Random,
    bg_color=(250, 248, 240),
) -> EvidenceAnnotation:
    draw = ImageDraw.Draw(img)
    pad = 3
    box = [fl.x - pad, fl.y - pad, fl.x + fl.w + pad, fl.y + fl.h + pad]

    if forgery_type == "text_replacement":
        draw.rectangle(box, fill=bg_color)
        new_text = {
            "name": _rand_name(),
            "dob": _rand_dob(),
            "id_number": _rand_id(),
            "address": _rand_address(),
        }[fl.name]
        font = ImageFont.truetype(FONT_REGULAR, 20)
        draw.text((fl.x, fl.y), new_text, font=font, fill=(10, 10, 15))  # subtly different black
        return EvidenceAnnotation(
            "color_and_noise_mismatch",
            "Replaced text region shows slightly different ink color and a flatter noise "
            "profile than the surrounding genuine text, consistent with digital re-insertion.",
        )

    if forgery_type == "font_modification":
        draw.rectangle(box, fill=bg_color)
        new_text = {
            "name": _rand_name(),
            "dob": _rand_dob(),
            "id_number": _rand_id(),
            "address": _rand_address(),
        }[fl.name]
        font = ImageFont.truetype(FONT_BOLD, 20)  # different weight than the rest of the doc
        draw.text((fl.x, fl.y), new_text, font=font, fill=(10, 10, 10))
        return EvidenceAnnotation(
            "font_mismatch",
            "Glyph stroke width and shape differ from the font used in every other field "
            "on the document, indicating the field was re-typed with a different font.",
        )

    if forgery_type == "spacing_modification":
        draw.rectangle(box, fill=bg_color)
        new_text = {
            "name": _rand_name(),
            "dob": _rand_dob(),
            "id_number": _rand_id(),
            "address": _rand_address(),
        }[fl.name]
        font = ImageFont.truetype(FONT_REGULAR, 20)
        x = fl.x
        for ch in new_text:
            draw.text((x, fl.y), ch, font=font, fill=(10, 10, 10))
            w = draw.textlength(ch, font=font)
            x += w + 4  # extra letter-spacing vs normal rendering
        return EvidenceAnnotation(
            "spacing_irregularity",
            "Character spacing (kerning) in this field is wider and less uniform than the "
            "document's normal typesetting, suggesting manual character-by-character editing.",
        )

    if forgery_type == "copy_paste":
        # copy a patch from elsewhere on the document (e.g. background area) over this field
        src_x, src_y = rng.randint(20, img.width - fl.w - 20), rng.randint(img.height - 60, img.height - 20)
        src_box = (src_x, src_y, src_x + fl.w, src_y + fl.h)
        patch = img.crop(src_box)
        img.paste(patch, (fl.x - pad, fl.y - pad))
        return EvidenceAnnotation(
            "edge_seam_discontinuity",
            "A visible rectangular seam and texture discontinuity at the field boundary "
            "indicates pixel content was copied from elsewhere and pasted into this region.",
        )

    raise ValueError(f"Unsupported text forgery_type: {forgery_type}")


def _tamper_photo_or_signature(
    img: Image.Image, fl: FieldLayout, rng: random.Random
) -> EvidenceAnnotation:
    draw = ImageDraw.Draw(img)
    if fl.kind == "photo":
        new_color = (rng.randint(20, 90), rng.randint(20, 90), rng.randint(90, 180))
        draw.rectangle([fl.x, fl.y, fl.x + fl.w, fl.y + fl.h], fill=(230, 230, 235), outline=(90, 90, 90), width=2)
        cx, cy = fl.x + fl.w // 2, fl.y + fl.h // 2 - 10
        draw.ellipse([cx - 30, cy - 40, cx + 30, cy + 30], fill=new_color)
        draw.ellipse([cx - 50, cy + 25, cx + 50, cy + 80], fill=new_color)
        return EvidenceAnnotation(
            "image_region_inconsistency",
            "The photo region has different color statistics and compression artifacts "
            "than the rest of the document, consistent with a swapped or replaced photo.",
        )
    else:
        draw.rectangle([fl.x, fl.y, fl.x + fl.w, fl.y + fl.h], fill=(255, 255, 255), outline=(150, 150, 150), width=1)
        _draw_signature(draw, fl, rng, color=(60, 20, 20))
        return EvidenceAnnotation(
            "signature_stroke_inconsistency",
            "Signature stroke dynamics (pressure/curvature proxy) differ from a natural "
            "single continuous signing motion, consistent with a redrawn signature.",
        )


def generate_document_group(
    group_id: str, img_w: int, img_h: int, out_dir: Path, rng: random.Random
) -> list[DocumentAnnotation]:
    """Generates ONE genuine document and ONE forged variant sharing `group_id`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    annotations = []

    # ---- genuine ----
    genuine_img, values, layouts = render_genuine_document(img_w, img_h, rng)
    genuine_path = out_dir / f"{group_id}_genuine.png"
    genuine_img.save(genuine_path)

    field_anns = [
        FieldAnnotation(
            field_name=fl.name,
            bbox=BoundingBox(fl.x, fl.y, fl.x + fl.w, fl.y + fl.h),
            is_tampered=False,
            forgery_type="none",
            evidence=[],
        )
        for fl in layouts
    ]
    annotations.append(
        DocumentAnnotation(
            document_id=f"{group_id}_genuine",
            document_group=group_id,
            image_path=str(genuine_path.name),
            image_width=img_w,
            image_height=img_h,
            is_forged=False,
            split="",  # assigned later at group level
            fields=field_anns,
            metadata={"values": values},
        )
    )

    # ---- forged variant ----
    forged_img, _, forged_layouts = render_genuine_document(img_w, img_h, rng)
    tamper_field = rng.choice(forged_layouts)
    text_forgeries = ["text_replacement", "font_modification", "spacing_modification", "copy_paste"]
    if tamper_field.kind == "text":
        forgery_type = rng.choice(text_forgeries)
        evidence = _tamper_text_field(forged_img, tamper_field, forgery_type, rng)
    else:
        forgery_type = "image_region_replacement"
        evidence = _tamper_photo_or_signature(forged_img, tamper_field, rng)

    forged_path = out_dir / f"{group_id}_forged.png"
    forged_img.save(forged_path)

    forged_field_anns = []
    for fl in forged_layouts:
        is_tampered = fl.name == tamper_field.name
        forged_field_anns.append(
            FieldAnnotation(
                field_name=fl.name,
                bbox=BoundingBox(fl.x, fl.y, fl.x + fl.w, fl.y + fl.h),
                is_tampered=is_tampered,
                forgery_type=forgery_type if is_tampered else "none",
                evidence=[evidence] if is_tampered else [],
            )
        )

    annotations.append(
        DocumentAnnotation(
            document_id=f"{group_id}_forged",
            document_group=group_id,
            image_path=str(forged_path.name),
            image_width=img_w,
            image_height=img_h,
            is_forged=True,
            split="",
            fields=forged_field_anns,
            metadata={"tampered_field": tamper_field.name, "forgery_type": forgery_type},
        )
    )

    return annotations
