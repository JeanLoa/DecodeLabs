"""Transform raw Titanic records into a model-ready table."""

import pandas as pd

from src.data import FEATURES, TARGET

NUMERIC = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL = ["Sex", "Embarked"]


def transform_titanic(raw: pd.DataFrame) -> pd.DataFrame:
    duplicate_key = ["PassengerId"] if "PassengerId" in raw.columns else None
    deduplicated = raw.drop_duplicates(subset=duplicate_key)
    data = deduplicated[[TARGET, *FEATURES]].copy().reset_index(drop=True)
    data[TARGET] = pd.to_numeric(data[TARGET], errors="raise").astype(int)
    for column in NUMERIC:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["Sex"] = data["Sex"].astype("string").str.strip().str.lower()
    data["Embarked"] = data["Embarked"].astype("string").str.strip().str.upper()
    if not set(data["Sex"].dropna().unique()).issubset({"female", "male"}):
        raise ValueError("Sex contains unsupported categories.")
    if not set(data["Embarked"].dropna().unique()).issubset({"C", "Q", "S"}):
        raise ValueError("Embarked contains unsupported categories.")
    for column in NUMERIC:
        data[column] = data[column].fillna(data[column].median())
    for column in CATEGORICAL:
        mode = data[column].mode(dropna=True)
        if mode.empty:
            raise ValueError(f"{column} has no usable values.")
        data[column] = data[column].fillna(mode.iloc[0])
    data["Pclass"] = data["Pclass"].astype(int)
    data["SibSp"] = data["SibSp"].astype(int)
    data["Parch"] = data["Parch"].astype(int)
    if not data["Pclass"].isin([1, 2, 3]).all():
        raise ValueError("Pclass must contain only 1, 2 or 3.")
    if data.isna().any().any():
        raise ValueError("Transformation left missing values in the model-ready dataset.")
    return data
