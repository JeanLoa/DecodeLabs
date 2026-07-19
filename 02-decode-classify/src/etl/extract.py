"""Extract Iris CSV data and normalize common source column names."""

from __future__ import annotations

from pathlib import Path
from typing import IO

import pandas as pd

from src.data import FEATURES, REQUIRED_COLUMNS, SPECIES, TARGET

_COLUMN_ALIASES = {
    "sepallength": FEATURES[0],
    "sepallengthcm": FEATURES[0],
    "sepalwidth": FEATURES[1],
    "sepalwidthcm": FEATURES[1],
    "petallength": FEATURES[2],
    "petallengthcm": FEATURES[2],
    "petalwidth": FEATURES[3],
    "petalwidthcm": FEATURES[3],
    "species": TARGET,
    "class": TARGET,
    "target": TARGET,
    "variety": TARGET,
}


def _column_key(column: object) -> str:
    return "".join(character for character in str(column).strip().lower() if character.isalnum())


def _canonicalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    renames: dict[object, str] = {}
    claimed: dict[str, object] = {}
    for column in data.columns:
        canonical = _COLUMN_ALIASES.get(_column_key(column))
        if canonical is None:
            continue
        if canonical in claimed:
            raise ValueError(
                f"Multiple source columns map to {canonical}: {claimed[canonical]!r} and {column!r}."
            )
        claimed[canonical] = column
        renames[column] = canonical
    return data.rename(columns=renames)


def extract_iris(source: str | Path | IO[bytes]) -> pd.DataFrame:
    """Read an Iris CSV and validate that its canonical schema is present."""
    data = _canonicalize_columns(pd.read_csv(source))
    if data.empty:
        raise ValueError("The uploaded Iris CSV is empty.")
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            "Expected four sepal/petal measurements and a species target."
        )
    return data


def detect_dataset_kind(data: pd.DataFrame) -> str:
    """Distinguish the canonical 150-row Iris dataset from custom uploads."""
    counts = data[TARGET].value_counts()
    if (
        len(data) == 150
        and set(counts.index) == set(SPECIES)
        and all(int(counts.get(species, 0)) == 50 for species in SPECIES)
    ):
        return "sklearn_iris"
    return "uploaded_iris"
