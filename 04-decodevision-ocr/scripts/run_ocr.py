"""Run DecodeVision against one image without Streamlit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision import (  # noqa: E402
    OCREngineUnavailable,
    PreprocessingConfig,
    TesseractEngine,
    decode_image,
    run_ocr_pipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract confidence-filtered text from one document image."
    )
    parser.add_argument("image", type=Path, help="Path to a PNG or JPG image.")
    parser.add_argument("--threshold", type=float, default=80.0)
    parser.add_argument("--psm", type=int, choices=[3, 6, 7, 11], default=6)
    parser.add_argument(
        "--threshold-mode",
        choices=["adaptive", "otsu"],
        default="adaptive",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        image = decode_image(args.image.read_bytes())
        result = run_ocr_pipeline(
            image,
            preprocessing_config=PreprocessingConfig(
                threshold_mode=args.threshold_mode
            ),
            confidence_threshold=args.threshold,
            page_segmentation_mode=args.psm,
            engine=TesseractEngine(),
        )
    except (OSError, ValueError, OCREngineUnavailable) as error:
        print(f"DecodeVision error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(result.recognition.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.recognition.passes_confidence_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
