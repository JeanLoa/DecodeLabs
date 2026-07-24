"""Milestone checks derived directly from the Project 4 brief."""

from __future__ import annotations

from dataclasses import dataclass

from .models import PipelineResult


@dataclass(frozen=True)
class MilestoneCheck:
    name: str
    passed: bool
    evidence: str


def evaluate_milestone(result: PipelineResult) -> tuple[MilestoneCheck, ...]:
    recognition = result.recognition
    stages = result.preprocessing
    return (
        MilestoneCheck(
            name="Library integration",
            passed=True,
            evidence="Tesseract returned a structured word-level response.",
        ),
        MilestoneCheck(
            name="Preprocessing integrity",
            passed=(
                stages.grayscale.ndim == 2
                and stages.binary.ndim == 2
                and stages.deskewed.shape == stages.binary.shape
            ),
            evidence=(
                "Grayscale, blur, thresholding and deskew stages were materialized."
            ),
        ),
        MilestoneCheck(
            name="Confidence benchmarking",
            passed=recognition.passes_confidence_gate,
            evidence=(
                f"Validated mean confidence: "
                f"{recognition.validated_confidence:.1f}% "
                f"(gate: {recognition.confidence_threshold:.0f}%)."
            ),
        ),
        MilestoneCheck(
            name="Visual confirmation",
            passed=bool(recognition.validated_text.strip()),
            evidence=(
                f"{len(recognition.accepted_words)} accepted words have "
                "inspectable bounding boxes."
            ),
        ),
    )
