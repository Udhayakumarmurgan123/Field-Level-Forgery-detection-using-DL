# Folder Structure

```
doc-forgery/
├── app.py                              # Gradio UI entry point
├── requirements.txt
├── README.md
├── configs/
│   └── config.yaml                     # single source of truth: fields, paths, hyperparams
│
├── src/
│   ├── data/
│   │   ├── schema.py                   # DocumentAnnotation / FieldAnnotation / EvidenceAnnotation dataclasses
│   │   ├── synthetic_generator.py      # genuine + forged ID-card document generator (ground-truth aware)
│   │   └── validator.py                # integrity, bbox, evidence, and leakage checks
│   │
│   ├── localization/
│   │   └── field_localizer.py          # inference-time YOLO wrapper (+ fallback heuristic layout)
│   │
│   ├── evidence_extraction/
│   │   └── classical_evidence.py       # ELA, noise-discontinuity, edge-seam, FFT-spacing signals
│   │
│   ├── forgery_detection/
│   │   └── field_classifier.py         # GradientBoosting classifier over evidence-signal features
│   │
│   ├── explanation/
│   │   └── template_explainer.py       # grounded NL explanation generator (VLM swap-in point)
│   │
│   ├── inference/
│   │   └── pipeline.py                 # ties every stage together end-to-end
│   │
│   └── ui/                             # (reserved for future UI helpers / VLM-serving code)
│
├── scripts/
│   ├── generate_synthetic_dataset.py   # step 1: build dataset + group-level split
│   ├── validate_dataset.py             # step 2: integrity + leakage check
│   ├── audit_synthetic_dataset.py      # step 3: class/field/forgery-type distributions
│   ├── visual_inspect_dataset.py       # step 4: render bboxes + labels for manual QA
│   ├── prepare_yolo_dataset.py         # step 5a: JSON -> YOLO detection format
│   ├── train_yolo.py                  # step 5b: train YOLOv8n field localizer
│   └── train_forgery_classifier.py     # step 6: build evidence features + train classifier
│
├── data/
│   ├── raw/
│   │   ├── annotations.json            # full ground-truth dataset (schema.py format)
│   │   └── images/                     # genuine_*.png / forged_*.png per document group
│   └── yolo/
│       ├── images/{train,val,test}/
│       ├── labels/{train,val,test}/    # YOLO-format .txt labels
│       └── data.yaml
│
├── outputs/
│   ├── checkpoints/
│   │   ├── field_localization_best.pt        # trained YOLO weights
│   │   ├── field_localization/                # full ultralytics training run (logs, plots)
│   │   └── field_forgery_classifier.joblib    # trained GradientBoosting classifier
│   ├── audit/
│   │   ├── audit_report.json
│   │   ├── audit_report.txt
│   │   └── visual_inspection/*.png
│   └── annotated/                       # annotated output images from live pipeline runs
│
├── docs/
│   ├── folder_structure.md              # this file
│   └── vlm_plan.md                      # plan for the future LoRA-VLM explanation upgrade
│
└── tests/                               # (reserved -- see "Suggested tests" in vlm_plan.md's sibling doc)
```

## Data flow summary

1. `generate_synthetic_dataset.py` writes `data/raw/annotations.json` + `data/raw/images/`.
2. `validate_dataset.py` / `audit_synthetic_dataset.py` / `visual_inspect_dataset.py` all read that JSON directly.
3. `prepare_yolo_dataset.py` converts it into `data/yolo/` (YOLO detection format), preserving the same train/val/test split.
4. `train_yolo.py` trains on `data/yolo/`, writes weights into `outputs/checkpoints/`.
5. `train_forgery_classifier.py` re-reads `data/raw/annotations.json` (ground-truth boxes, jittered) to build training features, and writes the classifier into `outputs/checkpoints/`.
6. `src/inference/pipeline.py` loads both trained artifacts from `outputs/checkpoints/` and runs the full chain on a new image.
7. `app.py` wraps `pipeline.py` in a Gradio UI.
