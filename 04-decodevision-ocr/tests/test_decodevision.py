"""Automated checks for DecodeVision's visual and OCR contracts."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.vision.models import PipelineResult, PreprocessingConfig
from src.vision.ocr import (
    discover_tesseract_command,
    parse_tesseract_data,
)
from src.vision.pipeline import draw_word_boxes, run_ocr_pipeline
from src.vision.preprocessing import InvalidImageError, decode_image, preprocess_image
from src.vision.validation import evaluate_milestone

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "samples" / "decodevision_invoice.png"


def tesseract_fixture() -> dict[str, list[object]]:
    return {
        "text": ["", "DECODEVISION", "LABS", "invoice", "noise"],
        "conf": [-1, 96.4, 92.1, 84.5, 42.0],
        "left": [0, 20, 205, 20, 400],
        "top": [0, 30, 30, 90, 90],
        "width": [0, 170, 70, 90, 60],
        "height": [0, 40, 40, 32, 32],
        "block_num": [0, 1, 1, 1, 1],
        "par_num": [0, 1, 1, 1, 1],
        "line_num": [0, 1, 1, 2, 2],
    }


class FakeEngine:
    def recognize(
        self,
        image: np.ndarray,
        *,
        confidence_threshold: float,
        page_segmentation_mode: int,
        language: str,
    ):
        self.image_shape = image.shape
        self.language = language
        return parse_tesseract_data(
            tesseract_fixture(),
            confidence_threshold=confidence_threshold,
            page_segmentation_mode=page_segmentation_mode,
        )


class ImageInputTests(unittest.TestCase):
    def test_versioned_sample_decodes_to_three_channels(self) -> None:
        image = decode_image(SAMPLE_PATH.read_bytes())

        self.assertEqual(image.ndim, 3)
        self.assertEqual(image.shape[2], 3)
        self.assertGreater(image.shape[0], 500)
        self.assertGreater(image.shape[1], 1000)

    def test_invalid_bytes_are_rejected(self) -> None:
        with self.assertRaises(InvalidImageError):
            decode_image(b"not-an-image")


class PreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = decode_image(SAMPLE_PATH.read_bytes())

    def test_all_preprocessing_stages_are_materialized(self) -> None:
        stages = preprocess_image(
            self.image,
            PreprocessingConfig(threshold_mode="adaptive", blur_kernel=3),
        )

        self.assertEqual(stages.original_rgb.shape, self.image.shape)
        self.assertEqual(stages.grayscale.ndim, 2)
        self.assertEqual(stages.blurred.shape, stages.grayscale.shape)
        self.assertEqual(stages.binary.shape, stages.grayscale.shape)
        self.assertEqual(stages.deskewed.shape, stages.binary.shape)
        self.assertTrue(set(np.unique(stages.binary)).issubset({0, 255}))

    def test_otsu_and_adaptive_thresholding_produce_binary_images(self) -> None:
        otsu = preprocess_image(
            self.image,
            PreprocessingConfig(threshold_mode="otsu"),
        )
        adaptive = preprocess_image(
            self.image,
            PreprocessingConfig(threshold_mode="adaptive"),
        )

        self.assertTrue(set(np.unique(otsu.binary)).issubset({0, 255}))
        self.assertTrue(set(np.unique(adaptive.binary)).issubset({0, 255}))
        self.assertFalse(np.array_equal(otsu.binary, adaptive.binary))

    def test_configuration_rejects_even_kernel_sizes(self) -> None:
        with self.assertRaises(ValueError):
            preprocess_image(
                self.image,
                PreprocessingConfig(blur_kernel=4),
            )

    def test_deskew_can_be_disabled_without_mutating_binary_stage(self) -> None:
        stages = preprocess_image(
            self.image,
            PreprocessingConfig(deskew=False),
        )

        self.assertEqual(stages.detected_angle, 0.0)
        np.testing.assert_array_equal(stages.binary, stages.deskewed)


class RecognitionTests(unittest.TestCase):
    def test_parser_rebuilds_lines_and_enforces_eighty_percent_gate(self) -> None:
        result = parse_tesseract_data(
            tesseract_fixture(),
            confidence_threshold=80.0,
            page_segmentation_mode=6,
        )

        self.assertEqual(len(result.words), 4)
        self.assertEqual(len(result.accepted_words), 3)
        self.assertEqual(result.raw_text, "DECODEVISION LABS\ninvoice noise")
        self.assertEqual(result.validated_text, "DECODEVISION LABS\ninvoice")
        self.assertGreaterEqual(result.validated_confidence, 80.0)
        self.assertTrue(result.passes_confidence_gate)

    def test_higher_gate_removes_words_without_changing_raw_evidence(self) -> None:
        result = parse_tesseract_data(
            tesseract_fixture(),
            confidence_threshold=95.0,
            page_segmentation_mode=6,
        )

        self.assertEqual(result.validated_text, "DECODEVISION")
        self.assertIn("noise", result.raw_text)
        self.assertEqual(result.accepted_ratio, 0.25)

    def test_pipeline_passes_deskewed_matrix_to_engine(self) -> None:
        image = decode_image(SAMPLE_PATH.read_bytes())
        engine = FakeEngine()

        result = run_ocr_pipeline(
            image,
            confidence_threshold=80,
            page_segmentation_mode=6,
            engine=engine,  # type: ignore[arg-type]
        )

        self.assertEqual(engine.image_shape, result.preprocessing.deskewed.shape)
        self.assertEqual(engine.language, "eng")
        self.assertTrue(result.recognition.passes_confidence_gate)

    def test_bounding_box_overlay_preserves_image_dimensions(self) -> None:
        image = decode_image(SAMPLE_PATH.read_bytes())
        engine = FakeEngine()
        result = run_ocr_pipeline(image, engine=engine)  # type: ignore[arg-type]

        overlay = draw_word_boxes(result)

        self.assertEqual(overlay.shape[:2], result.preprocessing.deskewed.shape)
        self.assertEqual(overlay.shape[2], 3)
        self.assertFalse(
            np.array_equal(
                overlay,
                cv2.cvtColor(result.preprocessing.deskewed, cv2.COLOR_GRAY2RGB),
            )
        )


class MilestoneTests(unittest.TestCase):
    def test_four_project_checks_pass_with_validated_output(self) -> None:
        image = decode_image(SAMPLE_PATH.read_bytes())
        result: PipelineResult = run_ocr_pipeline(
            image,
            engine=FakeEngine(),  # type: ignore[arg-type]
        )

        checks = evaluate_milestone(result)

        self.assertEqual(len(checks), 4)
        self.assertTrue(all(check.passed for check in checks))

    def test_explicit_tesseract_command_has_priority(self) -> None:
        with patch.dict(
            os.environ,
            {"TESSERACT_CMD": str(SAMPLE_PATH)},
            clear=False,
        ):
            self.assertEqual(discover_tesseract_command(), str(SAMPLE_PATH))


if __name__ == "__main__":
    unittest.main()
