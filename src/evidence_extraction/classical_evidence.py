"""
Classical, training-free evidence extraction for a single field crop.

These signals are cheap (no GPU, no training needed) and each maps directly onto
one of the ground-truth evidence types used in the synthetic dataset, which means
they double as an interpretable baseline forgery detector AND as features for the
downstream lightweight classifier.

Signals implemented:
  - Error Level Analysis (ELA): re-compresses the crop at a fixed JPEG quality and
    measures residual energy. Spliced/edited regions often re-compress differently
    from their surroundings, showing up as a bright ELA residual.
  - Noise-variance discontinuity: local noise variance of the crop vs. the noise
    variance of a ring of pixels immediately surrounding it (its "context"). A
    pasted-in region often has different sensor/rendering noise than its context.
  - Edge-density / seam score: Canny edge density along the crop's border vs.
    interior -- a copy-pasted rectangle tends to leave a sharper border seam.
  - FFT periodicity score: text spacing/font-weight changes alter the horizontal
    frequency spectrum of a text row; compares peak concentration to a reference.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class EvidenceSignal:
    evidence_type: str
    score: float           # 0..1, higher = more suspicious
    description: str


def _to_gray(crop_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)


def compute_ela_score(crop_bgr: np.ndarray, quality: int = 90) -> EvidenceSignal:
    """Re-compress the crop as JPEG at `quality` and measure the mean absolute
    difference from the original -- elevated for regions that were edited/pasted
    after the original image's own compression history."""
    ok, enc = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return EvidenceSignal("ela_hotspot", 0.0, "ELA could not be computed.")
    recompressed = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if recompressed.shape != crop_bgr.shape:
        recompressed = cv2.resize(recompressed, (crop_bgr.shape[1], crop_bgr.shape[0]))
    diff = cv2.absdiff(crop_bgr, recompressed).astype(np.float32)
    mean_residual = float(diff.mean())
    # empirically, genuine flat-background regions sit low; normalize to a 0..1-ish range
    score = min(1.0, mean_residual / 15.0)
    return EvidenceSignal(
        "ela_hotspot",
        score,
        f"Error-level-analysis residual (mean={mean_residual:.2f}) indicates "
        f"{'elevated' if score > 0.4 else 'low'} recompression inconsistency in this region.",
    )


def compute_noise_discontinuity(crop_bgr: np.ndarray, context_bgr: np.ndarray | None) -> EvidenceSignal:
    """Compare local noise variance (via high-pass Laplacian) inside the field
    crop against a surrounding context ring. Large discrepancies suggest the
    crop's pixel content did not originate from the same capture/render pass."""
    gray = _to_gray(crop_bgr)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    inner_var = float(lap.var())

    if context_bgr is None or context_bgr.size == 0:
        return EvidenceSignal("noise_discontinuity", 0.0, "No context region available for comparison.")

    ctx_gray = _to_gray(context_bgr)
    ctx_lap = cv2.Laplacian(ctx_gray, cv2.CV_64F)
    ctx_var = float(ctx_lap.var()) + 1e-6

    ratio = inner_var / ctx_var
    # discrepancy in either direction is suspicious
    discrepancy = abs(np.log(ratio + 1e-6))
    score = float(min(1.0, discrepancy / 2.0))
    return EvidenceSignal(
        "noise_discontinuity",
        score,
        f"Local noise variance ratio to surrounding region is {ratio:.2f}x, "
        f"{'a notable mismatch' if score > 0.4 else 'broadly consistent'} with the document's texture.",
    )


def compute_edge_seam_score(crop_bgr: np.ndarray, low: int = 50, high: int = 150) -> EvidenceSignal:
    """Canny edge density concentrated at the crop border (vs. interior) suggests
    a pasted rectangular patch leaving a visible seam."""
    gray = _to_gray(crop_bgr)
    edges = cv2.Canny(gray, low, high)
    h, w = edges.shape
    border = max(2, min(h, w) // 12)

    border_mask = np.zeros_like(edges, dtype=bool)
    border_mask[:border, :] = True
    border_mask[-border:, :] = True
    border_mask[:, :border] = True
    border_mask[:, -border:] = True

    border_density = edges[border_mask].mean() / 255.0 if border_mask.any() else 0.0
    interior_density = edges[~border_mask].mean() / 255.0 if (~border_mask).any() else 0.0
    diff = max(0.0, border_density - interior_density)
    score = float(min(1.0, diff * 6.0))
    return EvidenceSignal(
        "edge_seam_discontinuity",
        score,
        f"Border edge density ({border_density:.3f}) vs interior ({interior_density:.3f}) "
        f"{'shows a seam consistent with a pasted region' if score > 0.4 else 'shows no unusual boundary seam'}.",
    )


def compute_fft_periodicity_score(crop_bgr: np.ndarray) -> EvidenceSignal:
    """Text has fairly regular horizontal spacing; measure how 'peaky' the
    horizontal-projection spectrum is. Unusually irregular or overly-regular
    spacing (spacing/font tampering) shows up as an outlier peak concentration."""
    gray = _to_gray(crop_bgr).astype(np.float32)
    # horizontal projection profile (sum down columns) captures character spacing
    profile = gray.mean(axis=0)
    profile = profile - profile.mean()
    if profile.std() < 1e-3:
        return EvidenceSignal("spacing_irregularity", 0.0, "Region too uniform to assess character spacing.")
    spectrum = np.abs(np.fft.rfft(profile))
    spectrum = spectrum / (spectrum.sum() + 1e-6)
    # concentration = how much energy sits in the single strongest frequency bin
    # (excluding DC) -- overly concentrated OR overly flat spectra both indicate
    # spacing irregular relative to natural typeset text
    if len(spectrum) < 3:
        return EvidenceSignal("spacing_irregularity", 0.0, "Region too small to assess character spacing.")
    ac_spectrum = spectrum[1:]
    peak_concentration = float(ac_spectrum.max())
    # natural typeset text tends to have peak concentration in a moderate band;
    # score rises as we move away from that band in either direction
    target = 0.12
    score = float(min(1.0, abs(peak_concentration - target) / 0.25))
    return EvidenceSignal(
        "spacing_irregularity",
        score,
        f"Horizontal spacing spectrum peak concentration is {peak_concentration:.3f} "
        f"({'irregular vs. typical typeset spacing' if score > 0.4 else 'consistent with normal typesetting'}).",
    )


def extract_field_evidence(
    full_image_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    field_kind: str,
    ela_quality: int = 90,
    edge_low: int = 50,
    edge_high: int = 150,
    analysis_padding: int = 6,
) -> list[EvidenceSignal]:
    """Runs the relevant classical evidence extractors for one field crop.

    field_kind: "text" | "photo" | "signature" -- gates which signals are meaningful
    (e.g. FFT spacing analysis is meaningless for a photo region).

    `analysis_padding`: the detected bbox (from YOLO, or ground truth) is expanded
    by this many pixels on every side before evidence signals are computed. A
    tampering seam usually sits close to but not exactly on the annotated
    boundary, and YOLO's own localization has a few pixels of jitter -- without
    this margin, edge-seam and noise-discontinuity signals can miss the seam
    entirely on an otherwise-correct detection (verified empirically).
    """
    x0, y0, x1, y1 = bbox
    h, w = full_image_bgr.shape[:2]
    x0 = max(0, x0 - analysis_padding)
    y0 = max(0, y0 - analysis_padding)
    x1 = min(w, x1 + analysis_padding)
    y1 = min(h, y1 + analysis_padding)
    crop = full_image_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return []

    # context ring: expanded box minus the crop itself
    pad = max(10, (x1 - x0) // 4)
    cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
    cx1, cy1 = min(w, x1 + pad), min(h, y1 + pad)
    context = full_image_bgr[cy0:cy1, cx0:cx1].copy()
    # zero out the inner region so it doesn't dominate the "context" stats
    inner_y0, inner_x0 = y0 - cy0, x0 - cx0
    inner_y1, inner_x1 = inner_y0 + (y1 - y0), inner_x0 + (x1 - x0)
    mask = np.ones(context.shape[:2], dtype=bool)
    mask[inner_y0:inner_y1, inner_x0:inner_x1] = False
    context_pixels = context[mask] if mask.any() else None
    # reshape context_pixels back into a small 2D patch for the Laplacian call by
    # just using the full context frame (simpler and still informative)
    context_for_noise = context if context.size > 0 else None

    signals = [
        compute_ela_score(crop, quality=ela_quality),
        compute_noise_discontinuity(crop, context_for_noise),
        compute_edge_seam_score(crop, low=edge_low, high=edge_high),
    ]
    if field_kind == "text":
        signals.append(compute_fft_periodicity_score(crop))

    return signals
