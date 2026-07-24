"""Deterministic OpenCV preprocessing for document OCR."""

from __future__ import annotations

import cv2
import numpy as np

from .models import PreprocessedImage, PreprocessingConfig


class InvalidImageError(ValueError):
    """Raised when uploaded bytes cannot be decoded as an image."""


def decode_image(data: bytes) -> np.ndarray:
    """Decode image bytes into OpenCV BGR format."""

    if not data:
        raise InvalidImageError("The image payload is empty.")
    encoded = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise InvalidImageError("The file is not a valid supported image.")
    return image


def _threshold(gray: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    if config.threshold_mode == "otsu":
        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        return binary
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        config.adaptive_block_size,
        config.adaptive_constant,
    )


def deskew(binary: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotate foreground pixels onto a horizontal baseline."""

    foreground = np.column_stack(np.where(binary < 250))
    if len(foreground) < 20:
        return binary.copy(), 0.0

    raw_angle = float(cv2.minAreaRect(foreground)[-1])
    angle = -(90 + raw_angle) if raw_angle < -45 else -raw_angle
    if abs(angle) < 0.05 or abs(angle) > 15:
        return binary.copy(), 0.0

    height, width = binary.shape
    center = (width / 2.0, height / 2.0)
    rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        binary,
        rotation,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return rotated, angle


def preprocess_image(
    image_bgr: np.ndarray,
    config: PreprocessingConfig | None = None,
) -> PreprocessedImage:
    """Produce every visible preprocessing stage used by OCR."""

    active = config or PreprocessingConfig()
    active.validate()
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise InvalidImageError("Expected a three-channel BGR image.")

    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(
        grayscale,
        (active.blur_kernel, active.blur_kernel),
        0,
    )
    binary = _threshold(blurred, active)
    deskewed, angle = deskew(binary) if active.deskew else (binary.copy(), 0.0)
    return PreprocessedImage(
        original_rgb=original_rgb,
        grayscale=grayscale,
        blurred=blurred,
        binary=binary,
        deskewed=deskewed,
        detected_angle=angle,
    )
