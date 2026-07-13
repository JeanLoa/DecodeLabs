"""Extract and validate the raw Kaggle Titanic training dataset."""

from pathlib import Path
from typing import IO
from io import StringIO

import pandas as pd

from src.data import REQUIRED_COLUMNS, TARGET


def _repair_collapsed_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Repair CSVs whose complete rows were quoted into the first column."""
    if len(data.columns) < 2:
        return data
    trailing = data.iloc[:, 1:]
    first_column = data.iloc[:, 0].astype("string")
    if trailing.isna().all().all() and first_column.str.contains(",", regex=False).all():
        reconstructed = ",".join(str(column) for column in data.columns)
        reconstructed += "\n" + "\n".join(first_column.astype(str))
        return pd.read_csv(StringIO(reconstructed))
    return data


def detect_dataset_kind(data: pd.DataFrame) -> str:
    """Identify official training data or the labeled Kaggle test demo."""
    if (
        len(data) == 418
        and "PassengerId" in data.columns
        and pd.to_numeric(data["PassengerId"], errors="coerce").min() >= 892
    ):
        return "kaggle_test_with_submission_labels"
    return "kaggle_training"


def extract_titanic(source: str | Path | IO[bytes]) -> pd.DataFrame:
    data = _repair_collapsed_rows(pd.read_csv(source))
    if data.empty:
        raise ValueError("The uploaded CSV is empty.")
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        hint = " Kaggle test.csv has 418 rows and no Survived target; use train.csv instead." if len(data) == 418 else ""
        raise ValueError(f"Missing required columns: {missing}.{hint}")
    if data[TARGET].isna().any():
        raise ValueError("Survived contains missing labels. Use the Kaggle training dataset with complete targets.")
    targets = set(pd.to_numeric(data[TARGET], errors="coerce").dropna().unique())
    if targets != {0, 1}:
        raise ValueError("Survived must contain both binary classes: 0 and 1.")
    return data
