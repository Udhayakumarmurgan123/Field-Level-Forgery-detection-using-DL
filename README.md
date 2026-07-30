# Field-Level Explainable Document Forgery Detection

An end-to-end prototype that detects forged ID-style documents, localizes
**which field** was tampered, and explains **why**, using field-level
ground-truth evidence rather than a black-box genuine/forged label.

Built and tested end-to-end on CPU only (no GPU).

## Pipeline

```
Document Image
   -> Field Localization        (YOLOv8n, trained on synthetic data)
   -> Classical Evidence Extraction  (ELA, noise discontinuity, edge seam, FFT spacing)
   -> Lightweight Forgery Classifier (GradientBoosting on evidence-signal features)
   -> Template Explanation Generator (grounded NL explanation, no VLM required)
   -> Annotated image + structured JSON result
```

## Why this architecture (design rationale)

- **Synthetic data first.** ID-document forgery data with pixel-exact
  ground truth doesn't exist publicly at any scale, and real forged-ID data is
  legally/ethically sensitive to source. The synthetic generator
  (`src/data/synthetic_generator.py`) produces a genuine document and one
  forged variant per "document group", with **exact** bounding boxes and
  evidence descriptions, because we control the rendering.
- **Group-level train/val/test split.** A genuine document and its forged
  sibling share a `document_group` id and are always kept in the same split
  (enforced by `src/data/validator.py`) -- this is the single most common
  leakage bug in forgery-detection papers.
- **Classical evidence signals before a deep classifier.** ELA, noise-variance
  discontinuity, edge-seam density, and FFT spacing-periodicity are all
  training-free, CPU-instant, and directly interpretable -- each one *is* a
  piece of evidence shown to the end user, not just a black-box feature.
- **Lightweight classifier on top of evidence, not raw pixels.** A
  GradientBoostingClassifier over 4 evidence scores trains in seconds on CPU
  and keeps the whole reasoning chain inspectable: you can always ask "why did
  it flag this field?" and get the literal features that drove the decision.
- **Templated explanation instead of a VLM.** A LoRA-tuned VLM is the
  eventual goal (see `docs/vlm_plan.md`) but costs 10s of seconds per document
  on CPU and can hallucinate detail beyond what was measured. The template
  generator is 100% grounded in the same evidence shown in the UI, and is a
  drop-in replacement point once GPU is available.

## Results on this run

- **YOLOv8n field localization**: mAP50 = 0.994, mAP50-95 = 0.82 across all
  six field classes on the held-out validation split (120 document groups,
  96/11/13 train/val/test).
- **Forgery classifier**: ROC-AUC ≈ 0.96-0.99 on the held-out test split
  (numbers vary slightly run to run since the classifier is retrained here).

## An honest limitation found and partially fixed during development

The forgery classifier was originally trained and evaluated using
**ground-truth** bounding boxes, but at real inference time it runs on
**YOLO-predicted** boxes, which are accurate but not pixel-perfect. This
mismatch caused a real failure: a 2-pixel box shift changed the edge-seam
evidence score enough to flip a prediction. Two fixes were applied:

1. Training features are now built with **synthetic bbox jitter** (±4px) on
   the train split, so the classifier learns to tolerate the imprecision YOLO
   actually produces.
2. Evidence extraction now analyzes a **padded window** (+6px on each side)
   around the detected box, so seam/noise signals aren't lost to small
   misalignment.

This measurably improved robustness but did **not** fully close the gap --
recall on tampered fields in the held-out test set is currently ~0.6-1.0 run
to run, and a couple of the four evidence signals are more sensitive to this
kind of localization jitter than others (see feature importances printed by
`scripts/train_forgery_classifier.py`). This is flagged here rather than
hidden because it's the most important thing to improve next -- see below.

## What's next (not yet built)

1. **More evidence signals / a learned per-field CNN feature** to reduce
   reliance on the two jitter-sensitive classical signals (edge-seam,
   noise-discontinuity).
2. **Real (non-synthetic) validation images**, even a small hand-collected
   set, to check the synthetic-to-real domain gap.
3. **LoRA-tuned VLM explanation layer** as a v2 upgrade once GPU is available
   (see `docs/vlm_plan.md`) -- swap-in point is
   `src/explanation/template_explainer.py`.

## Repository layout

See `docs/folder_structure.md` for the full annotated tree.

## Running it

```bash
pip install -r requirements.txt

# 1. Generate the synthetic dataset (writes data/raw/annotations.json + images)
python scripts/generate_synthetic_dataset.py

# 2. Validate integrity + leakage
python scripts/validate_dataset.py

# 3. Audit distributions
python scripts/audit_synthetic_dataset.py

# 4. Visual spot-check
python scripts/visual_inspect_dataset.py --n 6

# 5. Convert to YOLO format and train field localization
python scripts/prepare_yolo_dataset.py
python scripts/train_yolo.py

# 6. Train the evidence-based forgery classifier
python scripts/train_forgery_classifier.py

# 7. Launch the app
python app.py
```
