"""Orchestrate the DecodeClassify ETL flow."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pandas as pd

from src.etl.extract import detect_dataset_kind, extract_titanic
from src.etl.load import save_processed, save_raw_upload
from src.etl.transform import transform_titanic


@dataclass(frozen=True)
class ETLResult:
    data: pd.DataFrame
    raw_rows: int
    processed_rows: int
    missing_before: int
    missing_after: int
    duplicates_removed: int
    output_path: Path
    source_kind: str
    warnings: tuple[str, ...]


def _warnings_for(source_kind: str) -> tuple[str, ...]:
    if source_kind == "kaggle_test_with_submission_labels":
        return (
            "Demo labels come from a Kaggle submission file, not verified training ground truth.",
        )
    return ()


def run_etl(raw_path: Path, processed_path: Path) -> ETLResult:
    raw = extract_titanic(raw_path)
    source_kind = detect_dataset_kind(raw)
    processed = transform_titanic(raw)
    save_processed(processed, processed_path)
    return ETLResult(
        data=processed,
        raw_rows=len(raw),
        processed_rows=len(processed),
        missing_before=int(raw.isna().sum().sum()),
        missing_after=int(processed.isna().sum().sum()),
        duplicates_removed=len(raw) - len(processed),
        output_path=processed_path,
        source_kind=source_kind,
        warnings=_warnings_for(source_kind),
    )


def run_uploaded_etl(
    content: bytes,
    raw_path: Path,
    processed_path: Path,
) -> ETLResult:
    """Validate an upload before preserving it as the official raw artifact."""
    raw = extract_titanic(BytesIO(content))
    source_kind = detect_dataset_kind(raw)
    processed = transform_titanic(raw)
    save_raw_upload(content, raw_path)
    save_processed(processed, processed_path)
    return ETLResult(
        data=processed,
        raw_rows=len(raw),
        processed_rows=len(processed),
        missing_before=int(raw.isna().sum().sum()),
        missing_after=int(processed.isna().sum().sum()),
        duplicates_removed=len(raw) - len(processed),
        output_path=processed_path,
        source_kind=source_kind,
        warnings=_warnings_for(source_kind),
    )
