"""Domain models for the OCR pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

ThresholdMode = Literal["otsu", "adaptive"]


@dataclass(frozen=True)
class PreprocessingConfig:
    """Parameters that shape visual input before recognition."""

    blur_kernel: int = 3
    threshold_mode: ThresholdMode = "adaptive"
    adaptive_block_size: int = 31
    adaptive_constant: int = 11
    deskew: bool = True

    def validate(self) -> None:
        if self.blur_kernel < 1 or self.blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be a positive odd number.")
        if self.adaptive_block_size < 3 or self.adaptive_block_size % 2 == 0:
            raise ValueError("adaptive_block_size must be an odd number >= 3.")
        if self.threshold_mode not in {"otsu", "adaptive"}:
            raise ValueError("threshold_mode must be 'otsu' or 'adaptive'.")


@dataclass(frozen=True)
class PreprocessedImage:
    """Inspectable stages produced by the IPO preprocessing pipeline."""

    original_rgb: np.ndarray
    grayscale: np.ndarray
    blurred: np.ndarray
    binary: np.ndarray
    deskewed: np.ndarray
    detected_angle: float


@dataclass(frozen=True)
class OCRWord:
    """One recognized word and its spatial evidence."""

    text: str
    confidence: float
    x: int
    y: int
    width: int
    height: int
    block: int
    paragraph: int
    line: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OCRResult:
    """Raw and confidence-filtered OCR output."""

    raw_text: str
    validated_text: str
    words: tuple[OCRWord, ...]
    accepted_words: tuple[OCRWord, ...]
    overall_confidence: float
    validated_confidence: float
    confidence_threshold: float
    page_segmentation_mode: int

    @property
    def passes_confidence_gate(self) -> bool:
        return bool(self.accepted_words) and (
            self.validated_confidence >= self.confidence_threshold
        )

    @property
    def accepted_ratio(self) -> float:
        return len(self.accepted_words) / len(self.words) if self.words else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "raw_text": self.raw_text,
            "validated_text": self.validated_text,
            "overall_confidence": round(self.overall_confidence, 2),
            "validated_confidence": round(self.validated_confidence, 2),
            "confidence_threshold": self.confidence_threshold,
            "page_segmentation_mode": self.page_segmentation_mode,
            "passes_confidence_gate": self.passes_confidence_gate,
            "accepted_ratio": round(self.accepted_ratio, 4),
            "words": [word.to_dict() for word in self.words],
            "accepted_words": [word.to_dict() for word in self.accepted_words],
        }


@dataclass(frozen=True)
class PipelineResult:
    """End-to-end output consumed by the interface and CLI."""

    preprocessing: PreprocessedImage
    recognition: OCRResult
