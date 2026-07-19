"""Transform extracted Iris records into the canonical model contract."""

from __future__ import annotations

import pandas as pd

from src.data import validate_dataset


def transform_iris(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize measurement types and species labels without leaking split data."""
    validated = validate_dataset(raw)
    return validated.drop_duplicates().reset_index(drop=True)
