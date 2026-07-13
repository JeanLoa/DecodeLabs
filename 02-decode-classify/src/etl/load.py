"""Persist raw and processed ETL artifacts."""

from pathlib import Path

import pandas as pd


def save_raw_upload(content: bytes, raw_path: Path) -> Path:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(content)
    return raw_path


def save_processed(data: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return output_path
