"""End-to-end orchestration and visual evidence helpers."""

from __future__ import annotations

import cv2
import numpy as np

from .models import PipelineResult, PreprocessingConfig
from .ocr import TesseractEngine
from .preprocessing import preprocess_image


def run_ocr_pipeline(
    image_bgr: np.ndarray,
    *,
    preprocessing_config: PreprocessingConfig | None = None,
    confidence_threshold: float = 80.0,
    page_segmentation_mode: int = 6,
    language: str = "eng",
    engine: TesseractEngine | None = None,
) -> PipelineResult:
    """Preprocess an image and recognize confidence-filtered text."""

    stages = preprocess_image(image_bgr, preprocessing_config)
    active_engine = engine or TesseractEngine()
    result = active_engine.recognize(
        stages.deskewed,
        confidence_threshold=confidence_threshold,
        page_segmentation_mode=page_segmentation_mode,
        language=language,
    )
    return PipelineResult(preprocessing=stages, recognition=result)


def draw_word_boxes(result: PipelineResult) -> np.ndarray:
    """Draw accepted and rejected OCR boxes over the deskewed input."""

    canvas = cv2.cvtColor(result.preprocessing.deskewed, cv2.COLOR_GRAY2RGB)
    accepted = set(result.recognition.accepted_words)
    for word in result.recognition.words:
        color = (15, 157, 88) if word in accepted else (217, 48, 37)
        start = (word.x, word.y)
        end = (word.x + word.width, word.y + word.height)
        cv2.rectangle(canvas, start, end, color, 2)
        label = f"{word.text} {word.confidence:.0f}%"
        cv2.putText(
            canvas,
            label,
            (word.x, max(14, word.y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas
