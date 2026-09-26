"""
Template-based, evidence-grounded natural language explanation generator.

Design rationale: a generative VLM (even with LoRA) is expensive to run on a
CPU-only laptop -- realistically 10s of seconds per document, and still prone to
hallucinating details not actually present in the evidence. A templated
generator conditioned directly on the *same* structured evidence signals shown
to the user in the UI gets most of the explainability value at a fraction of
the latency, is 100% grounded (it can never claim evidence that wasn't
measured), and gives predictable, review-able output for a security-sensitive
use case. This module is intentionally the whole "explanation" stage for the
CPU prototype; a LoRA-tuned VLM can later be swapped in behind the same
function signature once GPU is available (see docs/vlm_plan.md).
"""
from __future__ import annotations

from dataclasses import dataclass

from src.evidence_extraction.classical_evidence import EvidenceSignal

FIELD_DISPLAY_NAME = {
    "name": "Name",
    "dob": "Date of Birth",
    "id_number": "ID Number",
    "address": "Address",
    "photo": "Photo",
    "signature": "Signature",
}

FORGERY_TYPE_DISPLAY = {
    "text_replacement": "text replacement",
    "font_modification": "font modification",
    "spacing_modification": "spacing/kerning modification",
    "copy_paste": "copy-paste region manipulation",
    "image_region_replacement": "image region replacement",
    "none": "no manipulation",
}

# maps a dominant evidence_type -> a most-likely forgery_type label for display
# when we don't have a trained classifier for forgery *type* (only tampered/not)
EVIDENCE_TO_LIKELY_TYPE = {
    "ela_hotspot": "text_replacement",
    "noise_discontinuity": "image_region_replacement",
    "edge_seam_discontinuity": "copy_paste",
    "spacing_irregularity": "spacing_modification",
}


@dataclass
class FieldExplanation:
    field_name: str
    is_tampered: bool
    confidence: float
    likely_forgery_type: str
    explanation: str


def infer_likely_forgery_type(signals: list[EvidenceSignal]) -> str:
    if not signals:
        return "none"
    dominant = max(signals, key=lambda s: s.score)
    if dominant.score < 0.35:
        return "none"
    return EVIDENCE_TO_LIKELY_TYPE.get(dominant.evidence_type, "text_replacement")


def generate_field_explanation(
    field_name: str,
    is_tampered: bool,
    confidence: float,
    signals: list[EvidenceSignal],
) -> FieldExplanation:
    display = FIELD_DISPLAY_NAME.get(field_name, field_name)

    if not is_tampered:
        return FieldExplanation(
            field_name=field_name,
            is_tampered=False,
            confidence=confidence,
            likely_forgery_type="none",
            explanation=(
                f"{display} shows no significant evidence of tampering "
                f"(model confidence {confidence:.1%}). All measured signals "
                f"(compression consistency, noise texture, edge continuity) fall within "
                f"the expected range for an untouched field."
            ),
        )

    likely_type = infer_likely_forgery_type(signals)
    top_signals = sorted(signals, key=lambda s: s.score, reverse=True)[:2]
    signal_phrases = [f"{s.description}" for s in top_signals if s.score > 0.2]
    if not signal_phrases:
        signal_phrases = ["the combination of low-level pixel signals crossed the tampering threshold"]

    explanation = (
        f"{display} is flagged as tampered ({FORGERY_TYPE_DISPLAY.get(likely_type, likely_type)}, "
        f"model confidence {confidence:.1%}). Evidence: " + " ".join(signal_phrases)
    )

    return FieldExplanation(
        field_name=field_name,
        is_tampered=True,
        confidence=confidence,
        likely_forgery_type=likely_type,
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Helper utilities for document‑level summary (extracted to reduce cyclomatic
# complexity of the public function)
# ---------------------------------------------------------------------------

def _tampered_fields(field_explanations: list[FieldExplanation]) -> list[FieldExplanation]:
    """Return only the explanations that indicate tampering."""
    return [f for f in field_explanations if f.is_tampered]


def _joined_field_names(tampered: list[FieldExplanation]) -> str:
    """Create a comma‑separated string of display names for tampered fields."""
    return ", ".join(FIELD_DISPLAY_NAME.get(f.field_name, f.field_name) for f in tampered)


def _max_confidence(tampered: list[FieldExplanation]) -> float:
    """Return the highest confidence value among the tampered fields."""
    return max(f.confidence for f in tampered)


def generate_document_summary(field_explanations: list[FieldExplanation]) -> str:
    """Generate a concise summary describing the overall tampering status."""
    tampered = _tampered_fields(field_explanations)

    # Guard clause – no tampering detected
    if not tampered:
        return "No fields showed evidence of tampering. This document appears genuine."

    names = _joined_field_names(tampered)
    max_conf = _max_confidence(tampered)

    return (
        f"This document appears FORGED. {len(tampered)} field(s) flagged: {names}. "
        f"Highest-confidence finding: {max_conf:.1%}."
    )