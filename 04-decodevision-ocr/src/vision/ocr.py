"""Tesseract adapter and confidence-aware OCR result assembly."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from .models import OCRResult, OCRWord

WINDOWS_TESSERACT_LOCATIONS = (
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe",
)


class OCREngineUnavailable(RuntimeError):
    """Raised when Python or the native Tesseract engine is unavailable."""


def discover_tesseract_command() -> str | None:
    """Locate Tesseract from explicit configuration, PATH or common folders."""

    explicit = os.environ.get("TESSERACT_CMD", "").strip()
    if explicit and Path(explicit).is_file():
        return explicit
    from_path = shutil.which("tesseract")
    if from_path:
        return from_path
    for candidate in WINDOWS_TESSERACT_LOCATIONS:
        if candidate.is_file():
            return str(candidate)
    return None


def _column(data: Mapping[str, Sequence[Any]], name: str, index: int, default: Any) -> Any:
    values = data.get(name, ())
    return values[index] if index < len(values) else default


def parse_tesseract_data(
    data: Mapping[str, Sequence[Any]],
    *,
    confidence_threshold: float,
    page_segmentation_mode: int,
) -> OCRResult:
    """Convert pytesseract's columnar output into a stable domain contract."""

    texts = data.get("text", ())
    words: list[OCRWord] = []
    for index, raw_text in enumerate(texts):
        text = str(raw_text).strip()
        if not text:
            continue
        try:
            confidence = float(_column(data, "conf", index, -1))
        except (TypeError, ValueError):
            confidence = -1.0
        if confidence < 0:
            continue
        words.append(
            OCRWord(
                text=text,
                confidence=max(0.0, min(100.0, confidence)),
                x=int(_column(data, "left", index, 0)),
                y=int(_column(data, "top", index, 0)),
                width=int(_column(data, "width", index, 0)),
                height=int(_column(data, "height", index, 0)),
                block=int(_column(data, "block_num", index, 0)),
                paragraph=int(_column(data, "par_num", index, 0)),
                line=int(_column(data, "line_num", index, 0)),
            )
        )

    accepted = tuple(
        word for word in words if word.confidence >= confidence_threshold
    )
    return OCRResult(
        raw_text=_rebuild_lines(tuple(words)),
        validated_text=_rebuild_lines(accepted),
        words=tuple(words),
        accepted_words=accepted,
        overall_confidence=fmean(word.confidence for word in words) if words else 0.0,
        validated_confidence=(
            fmean(word.confidence for word in accepted) if accepted else 0.0
        ),
        confidence_threshold=confidence_threshold,
        page_segmentation_mode=page_segmentation_mode,
    )


def _rebuild_lines(words: tuple[OCRWord, ...]) -> str:
    lines: list[str] = []
    current_key: tuple[int, int, int] | None = None
    current_words: list[str] = []
    for word in words:
        key = (word.block, word.paragraph, word.line)
        if current_key is not None and key != current_key:
            lines.append(" ".join(current_words))
            current_words = []
        current_key = key
        current_words.append(word.text)
    if current_words:
        lines.append(" ".join(current_words))
    return "\n".join(lines)


class TesseractEngine:
    """Thin native-library boundary around pytesseract."""

    def __init__(self, command: str | None = None) -> None:
        self.command = command or discover_tesseract_command()

    @property
    def available(self) -> bool:
        return bool(self.command)

    def recognize(
        self,
        image: np.ndarray,
        *,
        confidence_threshold: float = 80.0,
        page_segmentation_mode: int = 6,
        language: str = "eng",
    ) -> OCRResult:
        if not 0 <= confidence_threshold <= 100:
            raise ValueError("confidence_threshold must be between 0 and 100.")
        if page_segmentation_mode not in {3, 6, 7, 11}:
            raise ValueError("Unsupported page segmentation mode.")
        if not self.command:
            raise OCREngineUnavailable(
                "Tesseract OCR is not installed or TESSERACT_CMD is not configured."
            )
        try:
            import pytesseract
            from pytesseract import Output
        except ImportError as error:
            raise OCREngineUnavailable(
                "The pytesseract Python package is not installed."
            ) from error

        pytesseract.pytesseract.tesseract_cmd = self.command
        try:
            data = pytesseract.image_to_data(
                image,
                lang=language,
                config=f"--oem 3 --psm {page_segmentation_mode}",
                output_type=Output.DICT,
            )
        except pytesseract.TesseractError as error:
            raise OCREngineUnavailable(f"Tesseract failed: {error}") from error
        return parse_tesseract_data(
            data,
            confidence_threshold=confidence_threshold,
            page_segmentation_mode=page_segmentation_mode,
        )
