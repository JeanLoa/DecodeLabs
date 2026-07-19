"""Orchestrate the DecodeClassify Iris ETL flow."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pandas as pd

from src.data import REQUIRED_COLUMNS, SPECIES, TARGET
from src.etl.extract import detect_dataset_kind, extract_iris
from src.etl.load import save_processed, save_raw_upload
from src.etl.transform import transform_iris


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
    class_counts: dict[str, int]
    warnings: tuple[str, ...]


def _build_result(raw: pd.DataFrame, processed: pd.DataFrame, output_path: Path) -> ETLResult:
    counts = processed[TARGET].value_counts().reindex(SPECIES, fill_value=0)
    return ETLResult(
        data=processed,
        raw_rows=len(raw),
        processed_rows=len(processed),
        missing_before=int(raw.loc[:, list(REQUIRED_COLUMNS)].isna().sum().sum()),
        missing_after=int(processed.loc[:, list(REQUIRED_COLUMNS)].isna().sum().sum()),
        duplicates_removed=len(raw) - len(processed),
        output_path=output_path,
        source_kind=detect_dataset_kind(processed),
        class_counts={species: int(counts[species]) for species in SPECIES},
        warnings=(),
    )


def run_etl(raw_path: Path, processed_path: Path) -> ETLResult:
    """Extract, validate and load an Iris CSV from disk."""
    raw = extract_iris(raw_path)
    processed = transform_iris(raw)
    save_processed(processed, processed_path)
    return _build_result(raw, processed, processed_path)


def run_uploaded_etl(
    content: bytes,
    raw_path: Path,
    processed_path: Path,
) -> ETLResult:
    """Validate an upload before preserving it as the raw source."""
    raw = extract_iris(BytesIO(content))
    processed = transform_iris(raw)
    save_raw_upload(content, raw_path)
    save_processed(processed, processed_path)
    return _build_result(raw, processed, processed_path)
