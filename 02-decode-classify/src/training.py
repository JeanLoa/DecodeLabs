"""K-Nearest Neighbors training, evaluation and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import FEATURES, SPECIES, TARGET, validate_dataset


@dataclass(frozen=True)
class TrainingResult:
    pipeline: Pipeline
    accuracy: float
    macro_f1: float
    weighted_f1: float
    confusion: list[list[int]]
    report: dict
    train_rows: int
    test_rows: int
    class_names: tuple[str, ...]


def _validate_neighbors(neighbors: int) -> None:
    if isinstance(neighbors, bool) or not isinstance(neighbors, int) or neighbors < 1:
        raise ValueError("neighbors must be a positive integer.")


def _validate_training_data(data: pd.DataFrame) -> pd.DataFrame:
    validated = validate_dataset(data)
    counts = validated[TARGET].value_counts()
    if int(counts.min()) < 2:
        raise ValueError("Each Iris species needs at least two rows for a stratified split.")
    return validated


def build_pipeline(neighbors: int = 5) -> Pipeline:
    """Create a leakage-safe scaler and KNN classifier pipeline."""
    _validate_neighbors(neighbors)
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                KNeighborsClassifier(n_neighbors=neighbors, n_jobs=1),
            ),
        ]
    )


def train_classifier(
    data: pd.DataFrame,
    neighbors: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainingResult:
    """Train and evaluate KNN using a reproducible stratified holdout."""
    _validate_neighbors(neighbors)
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    validated = _validate_training_data(data)
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            validated.loc[:, list(FEATURES)],
            validated[TARGET],
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=validated[TARGET],
        )
    except ValueError as error:
        raise ValueError(f"Unable to create a stratified train/test split: {error}") from error

    if neighbors > len(x_train):
        raise ValueError(
            f"neighbors ({neighbors}) cannot exceed the number of training rows ({len(x_train)})."
        )

    pipeline = build_pipeline(neighbors)
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    labels = list(SPECIES)
    return TrainingResult(
        pipeline=pipeline,
        accuracy=float(accuracy_score(y_test, predictions)),
        macro_f1=float(f1_score(y_test, predictions, labels=labels, average="macro", zero_division=0)),
        weighted_f1=float(
            f1_score(y_test, predictions, labels=labels, average="weighted", zero_division=0)
        ),
        confusion=confusion_matrix(y_test, predictions, labels=labels).tolist(),
        report=classification_report(
            y_test,
            predictions,
            labels=labels,
            target_names=labels,
            output_dict=True,
            zero_division=0,
        ),
        train_rows=len(x_train),
        test_rows=len(x_test),
        class_names=SPECIES,
    )


def predict_flower(
    result: TrainingResult,
    flower: Mapping[str, float],
) -> tuple[str, float, dict[str, float]]:
    """Predict one Iris species and return confidence plus every class probability."""
    missing = [feature for feature in FEATURES if feature not in flower]
    if missing:
        raise ValueError(f"Missing flower measurements: {missing}")

    values: dict[str, float] = {}
    for feature in FEATURES:
        try:
            measurement = float(flower[feature])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{feature} must be numeric.") from error
        if not np.isfinite(measurement) or measurement <= 0:
            raise ValueError(f"{feature} must be a finite measurement greater than zero.")
        values[feature] = measurement

    sample = pd.DataFrame([values], columns=list(FEATURES))
    prediction = str(result.pipeline.predict(sample)[0])
    classifier = result.pipeline.named_steps["classifier"]
    probabilities_array = result.pipeline.predict_proba(sample)[0]
    probabilities = {
        str(species): float(probability)
        for species, probability in zip(classifier.classes_, probabilities_array)
    }
    return prediction, probabilities[prediction], probabilities


def tune_neighbors(
    data: pd.DataFrame,
    max_neighbors: int = 15,
    folds: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Tune K on training folds while preserving an untouched holdout."""
    _validate_neighbors(max_neighbors)
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError("folds must be an integer of at least 2.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    validated = _validate_training_data(data)
    try:
        x_train, _, y_train, _ = train_test_split(
            validated.loc[:, list(FEATURES)],
            validated[TARGET],
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=validated[TARGET],
        )
    except ValueError as error:
        raise ValueError(
            f"Unable to preserve a stratified holdout before tuning: {error}"
        ) from error
    smallest_class = int(y_train.value_counts().min())
    if folds > smallest_class:
        raise ValueError(
            "folds cannot exceed the smallest training class size "
            f"({smallest_class}); received {folds}."
        )

    cross_validation = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )
    minimum_training_rows = min(
        len(train_indices)
        for train_indices, _ in cross_validation.split(x_train, y_train)
    )
    upper_bound = min(max_neighbors, minimum_training_rows)

    rows: list[dict[str, float | int]] = []
    for neighbors in range(1, upper_bound + 1):
        scores: list[float] = []
        for fit_indices, validation_indices in cross_validation.split(
            x_train,
            y_train,
        ):
            pipeline = build_pipeline(neighbors)
            pipeline.fit(
                x_train.iloc[fit_indices],
                y_train.iloc[fit_indices],
            )
            predictions = pipeline.predict(x_train.iloc[validation_indices])
            scores.append(
                float(
                    accuracy_score(
                        y_train.iloc[validation_indices],
                        predictions,
                    )
                )
            )
        score_array = np.asarray(scores, dtype=float)
        rows.append(
            {
                "neighbors": neighbors,
                "mean_accuracy": float(score_array.mean()),
                "std_accuracy": float(score_array.std(ddof=0)),
            }
        )
    return pd.DataFrame(rows, columns=["neighbors", "mean_accuracy", "std_accuracy"])
