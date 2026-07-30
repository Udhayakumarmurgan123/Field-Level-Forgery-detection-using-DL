# VLM + LoRA Explanation Layer -- Future Plan

This documents how to upgrade `src/explanation/template_explainer.py` to a
generative Vision-Language Model once GPU compute is available, without
changing anything else in the pipeline.

## Why this wasn't built first (CPU constraint)

Even a small (3-4B parameter) VLM takes roughly 10-60+ seconds per document on
CPU-only inference -- unacceptable for an interactive Gradio demo. The
template-based explainer (already built) is grounded 100% in the same
evidence signals the classifier used, so it cannot hallucinate a claim that
wasn't actually measured. This makes it a legitimate and defensible v1 for a
security-sensitive use case, not just a placeholder.

## Swap-in point

Replace the internals of `generate_field_explanation()` in
`src/explanation/template_explainer.py`. The function signature should stay
the same so `src/inference/pipeline.py` needs no changes:

```python
def generate_field_explanation(
    field_name: str,
    is_tampered: bool,
    confidence: float,
    signals: list[EvidenceSignal],
) -> FieldExplanation:
    ...
```

## Planned approach

1. **Base model**: a small open vision-language model (e.g. in the 2-4B
   parameter range) that accepts an image crop + text prompt.
2. **LoRA fine-tuning target**: condition the model on (a) the field crop
   image, (b) the structured evidence signals (as text, exactly as already
   shown in the UI), and train it to produce a one-paragraph, evidence-cited
   explanation. Using the *same* structured evidence as conditioning (rather
   than asking the VLM to detect tampering itself) keeps the model's job
   narrowly "explain what these measurements imply" rather than "detect
   forgery from pixels", which is both an easier learning problem and keeps
   the explanation grounded.
3. **Training data**: the synthetic dataset's `EvidenceAnnotation.description`
   fields are exactly the target explanations to imitate style-wise; augment
   with paraphrases for diversity.
4. **Fallback behavior**: keep the template explainer as an automatic fallback
   if VLM inference exceeds a latency budget (e.g. 5s) or fails to load, so
   the app remains usable on CPU-only machines.

## Suggested next experiment before committing to this

Before investing in LoRA fine-tuning, run a quick prompt-only baseline: feed a
frozen open VLM the field crop + evidence signals with a zero-shot prompt and
manually inspect 20-30 outputs for hallucination rate and latency. If
zero-shot quality is already acceptable, LoRA fine-tuning may only be needed
for latency/cost reasons (a smaller fine-tuned model matching a larger
zero-shot model's quality), not quality reasons.
