from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image

from src.inference.pipeline import PrototypeDocumentAnalysisPipeline

pipeline = PrototypeDocumentAnalysisPipeline()


def _resolve_input_path(image_file: Any, tmpdir: Path) -> Path:
    """Handles whatever Gradio hands us -- a plain filepath string (expected,
    since the component below uses type='filepath'), but also numpy arrays,
    PIL Images, or dict/object FileData wrappers, in case Gradio's internal
    representation changes again across versions."""
    if isinstance(image_file, str):
        return Path(image_file)
    if isinstance(image_file, Path):
        return image_file
    if isinstance(image_file, np.ndarray):
        p = tmpdir / "uploaded_document.png"
        Image.fromarray(image_file).save(p)
        return p
    if isinstance(image_file, Image.Image):
        p = tmpdir / "uploaded_document.png"
        image_file.save(p)
        return p
    # dict or SimpleNamespace-style FileData wrapper
    path = None
    if isinstance(image_file, dict):
        path = image_file.get("path") or image_file.get("url")
    else:
        path = getattr(image_file, "path", None) or getattr(image_file, "url", None)
    if path:
        return Path(path)
    raise gr.Error(f"Unsupported image input type from Gradio: {type(image_file)}")


def analyze_document(image_file) -> tuple[gr.components.Image, str, str, str, str, str]:
    if image_file is None:
        raise gr.Error("Please upload an image first.")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = _resolve_input_path(image_file, Path(tmpdir))
        result = pipeline.analyze_image(temp_path)

    annotated_path = Path(result["annotated_image_path"])
    summary = [
        f"Overall status: {result['overall_status']}",
        f"Overall confidence: {result['overall_confidence']:.4f}",
        f"Mode: {result['mode']}",
        f"Note: {result['note']}",
    ]

    field_rows = []
    for item in result["field_results"]:
        field_rows.append(
            f"- {item['field_name']}: status={item['tampering_status']} | forgery_type={item['forgery_type']} | confidence={item['confidence']:.3f}"
        )

    field_table = "\n".join(field_rows)
    evidence_table = ""
    for item in result["field_results"]:
        evidence_table += f"Field {item['field_name']}\n"
        for evidence in item["evidence"]:
            evidence_table += f"  • {evidence['type']}: {evidence['confidence']:.4f} - {evidence['description']}\n"
        evidence_table += "\n"

    explanation_text = "\n\n".join(
        f"{item['field_name']}: {item['explanation']}"
        for item in result["field_results"]
    )

    return (
        gr.Image(value=str(annotated_path)),
        "\n".join(summary),
        field_table,
        evidence_table,
        result.get("tampered_field") or "None",
        explanation_text,
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Field-Level Explainable Document Forgery Detection MVP") as demo:
        gr.Markdown(
            "# Field-Level Explainable Document Forgery Detection\n"
            "Prototype analysis mode for CPU-friendly local inspection."
        )
        with gr.Row():
            input_image = gr.Image(type="filepath", label="Upload document image")
            output_image = gr.Image(label="Annotated document")
        with gr.Row():
            run_button = gr.Button("Analyze document")
        with gr.Row():
            summary_box = gr.Textbox(label="Overall result", lines=8)
        with gr.Tab("Field results"):
            field_box = gr.Textbox(label="Field-level outcomes", lines=12)
        with gr.Tab("Evidence"):
            evidence_box = gr.Textbox(label="Visual evidence", lines=16)
        with gr.Tab("Explanation"):
            tampered_box = gr.Textbox(label="Tampered field")
            explanation_box = gr.Textbox(label="Grounded explanation", lines=12)

        run_button.click(
            fn=analyze_document,
            inputs=[input_image],
            outputs=[output_image, summary_box, field_box, evidence_box, tampered_box, explanation_box],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7862, share=False)