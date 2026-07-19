"""Command-line entry point for the DecodeClassify Iris ETL pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.etl import run_etl  # noqa: E402


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_report(status: str, source: Path, output: Path, detail: dict) -> Path:
    report_path = PROJECT_ROOT / "data" / "processed" / "etl_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": status,
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "source": _display_path(source),
        "output": _display_path(output),
        **detail,
    }
    temporary_path = report_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary_path.replace(report_path)
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Iris raw-to-processed ETL")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data/raw/iris.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/iris_processed.csv",
    )
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output.resolve()

    try:
        result = run_etl(source, output)
    except Exception as error:
        report = write_report(
            "rejected",
            source,
            output,
            {"error_type": type(error).__name__, "error": str(error)},
        )
        print(f"ETL rejected: {error}")
        print(f"Quality report: {report}")
        return 1

    report = write_report(
        "completed",
        source,
        output,
        {
            "raw_rows": result.raw_rows,
            "processed_rows": result.processed_rows,
            "missing_before": result.missing_before,
            "missing_after": result.missing_after,
            "duplicates_removed": result.duplicates_removed,
            "source_kind": result.source_kind,
            "class_counts": result.class_counts,
            "warnings": list(result.warnings),
        },
    )
    print("ETL completed")
    print(f"Raw rows: {result.raw_rows}")
    print(f"Processed rows: {result.processed_rows}")
    print(f"Missing values: {result.missing_before} -> {result.missing_after}")
    print(f"Source kind: {result.source_kind}")
    print(f"Class counts: {result.class_counts}")
    for warning in result.warnings:
        print(f"Warning: {warning}")
    print(f"Output: {result.output_path}")
    print(f"Quality report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
