"""Computer-vision building blocks for DecodeVision."""

from .models import OCRResult, OCRWord, PipelineResult, PreprocessedImage, PreprocessingConfig
from .ocr import OCREngineUnavailable, TesseractEngine
from .pipeline import draw_word_boxes, run_ocr_pipeline
from .preprocessing import decode_image, preprocess_image

__all__ = [
    "OCREngineUnavailable",
    "OCRResult",
    "OCRWord",
    "PipelineResult",
    "PreprocessedImage",
    "PreprocessingConfig",
    "TesseractEngine",
    "decode_image",
    "draw_word_boxes",
    "preprocess_image",
    "run_ocr_pipeline",
]
