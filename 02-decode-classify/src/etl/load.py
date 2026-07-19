"""Persist raw and processed Iris ETL artifacts atomically."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd


def _atomic_bytes_write(content: bytes, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def save_raw_upload(content: bytes, raw_path: Path) -> Path:
    """Preserve a validated upload exactly as received."""
    return _atomic_bytes_write(content, raw_path)


def save_processed(data: pd.DataFrame, output_path: Path) -> Path:
    """Write the canonical processed CSV without exposing partial output."""
    csv_content = data.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return _atomic_bytes_write(csv_content, output_path)
