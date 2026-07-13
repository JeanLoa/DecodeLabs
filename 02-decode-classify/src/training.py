"""Classification training pipeline."""

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data import FEATURES, TARGET

NUMERIC = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL = ["Sex", "Embarked"]


@dataclass
class TrainingResult:
    pipeline: Pipeline
    accuracy: float
    confusion: list[list[int]]
    report: dict
    train_rows: int
    test_rows: int


def build_pipeline(algorithm: str, max_depth: int = 4) -> Pipeline:
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]
    )
    processor = ColumnTransformer(
        [("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)]
    )
    if algorithm == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    else:
        model = LogisticRegression(max_iter=1000, random_state=42)
    return Pipeline([("processor", processor), ("model", model)])


def train_classifier(
    data: pd.DataFrame,
    algorithm: str = "Logistic Regression",
    test_size: float = 0.2,
    max_depth: int = 4,
) -> TrainingResult:
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES], data[TARGET], test_size=test_size, random_state=42, stratify=data[TARGET]
    )
    pipeline = build_pipeline(algorithm, max_depth)
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    return TrainingResult(
        pipeline=pipeline,
        accuracy=float(accuracy_score(y_test, predictions)),
        confusion=confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        report=classification_report(y_test, predictions, output_dict=True, zero_division=0),
        train_rows=len(x_train),
        test_rows=len(x_test),
    )


def predict_passenger(result: TrainingResult, passenger: dict) -> tuple[int, float]:
    sample = pd.DataFrame([passenger], columns=FEATURES)
    prediction = int(result.pipeline.predict(sample)[0])
    probability = float(result.pipeline.predict_proba(sample)[0][prediction])
    return prediction, probability
