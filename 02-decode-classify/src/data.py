"""Titanic dataset loading and validation."""

from pathlib import Path
from typing import BinaryIO

import pandas as pd

TARGET = "Survived"
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
REQUIRED_COLUMNS = [TARGET, *FEATURES]


def load_dataset(source: str | Path | BinaryIO) -> pd.DataFrame:
    data = pd.read_csv(source)
    if data.empty:
        raise ValueError("The dataset is empty.")
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if data[TARGET].isna().any():
        raise ValueError("Survived contains missing labels.")
    invalid_targets = set(data[TARGET].dropna().unique()).difference({0, 1})
    if invalid_targets:
        raise ValueError("Survived must contain only 0 and 1.")
    return data


def dataset_summary(data: pd.DataFrame) -> dict[str, int | float]:
    return {
        "rows": len(data),
        "features": len(FEATURES),
        "missing": int(data[REQUIRED_COLUMNS].isna().sum().sum()),
        "survival_rate": float(data[TARGET].mean()),
    }
