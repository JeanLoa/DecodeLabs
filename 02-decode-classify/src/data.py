"""Canonical Iris dataset contract and loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

FEATURES = (
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
)
TARGET = "species"
SPECIES = ("setosa", "versicolor", "virginica")
REQUIRED_COLUMNS = (*FEATURES, TARGET)


def _normalize_species(series: pd.Series) -> pd.Series:
    aliases = {
        "0": "setosa",
        "1": "versicolor",
        "2": "virginica",
        "iris-setosa": "setosa",
        "iris-versicolor": "versicolor",
        "iris-virginica": "virginica",
    }
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.replace(aliases)


def validate_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Return a validated canonical copy of an Iris classification dataset."""
    if data.empty:
        raise ValueError("The Iris dataset is empty.")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    validated = data.loc[:, list(REQUIRED_COLUMNS)].copy()
    for column in FEATURES:
        source = validated[column]
        numeric = pd.to_numeric(source, errors="coerce")
        if numeric.isna().any():
            raise ValueError(f"{column} must contain complete numeric measurements.")
        if not np.isfinite(numeric).all():
            raise ValueError(f"{column} must contain only finite measurements.")
        if (numeric <= 0).any():
            raise ValueError(f"{column} must contain measurements greater than zero.")
        validated[column] = numeric.astype(float)

    if validated[TARGET].isna().any():
        raise ValueError("species contains missing labels.")
    validated[TARGET] = _normalize_species(validated[TARGET])

    observed = set(validated[TARGET].dropna().unique())
    unsupported = observed.difference(SPECIES)
    if unsupported:
        raise ValueError(f"species contains unsupported labels: {sorted(unsupported)}")
    missing_species = set(SPECIES).difference(observed)
    if missing_species:
        raise ValueError(f"species must contain all three Iris classes; missing: {sorted(missing_species)}")

    return validated.reset_index(drop=True)


def load_dataset(source: str | Path | BinaryIO) -> pd.DataFrame:
    """Load a processed Iris CSV and enforce the model contract."""
    return validate_dataset(pd.read_csv(source))


def dataset_summary(data: pd.DataFrame) -> dict[str, int | dict[str, int]]:
    """Summarize rows, features, missing values and class balance."""
    validated = validate_dataset(data)
    counts = validated[TARGET].value_counts().reindex(SPECIES, fill_value=0)
    return {
        "rows": len(validated),
        "features": len(FEATURES),
        "classes": len(SPECIES),
        "missing": int(validated.loc[:, list(REQUIRED_COLUMNS)].isna().sum().sum()),
        "class_counts": {species: int(counts[species]) for species in SPECIES},
    }
