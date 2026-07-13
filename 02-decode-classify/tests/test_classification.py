import unittest
from io import StringIO
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.data import dataset_summary, load_dataset
from src.training import predict_passenger, train_classifier
from src.etl import run_uploaded_etl
from src.etl.extract import extract_titanic
from src.etl.transform import transform_titanic


def sample_data() -> pd.DataFrame:
    rows = []
    for index in range(40):
        rows.append({"Survived": index % 2, "Pclass": 1 if index % 2 else 3, "Sex": "female" if index % 2 else "male", "Age": 20 + index, "SibSp": index % 3, "Parch": 0, "Fare": 20.0 + index, "Embarked": "S" if index % 3 else "C"})
    return pd.DataFrame(rows)


class ClassificationTests(unittest.TestCase):
    def setUp(self):
        self.data = sample_data()

    def test_loads_and_summarizes_dataset(self):
        source = StringIO(self.data.to_csv(index=False))
        loaded = load_dataset(source)
        self.assertEqual(dataset_summary(loaded)["rows"], 40)

    def test_rejects_missing_columns(self):
        source = StringIO(pd.DataFrame({"Survived": [0, 1]}).to_csv(index=False))
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            load_dataset(source)

    def test_rejects_kaggle_test_file_without_target(self):
        test_data = self.data.drop(columns=["Survived"])
        test_data = pd.concat([test_data] * 11, ignore_index=True).head(418)
        with self.assertRaisesRegex(ValueError, "test.csv"):
            extract_titanic(StringIO(test_data.to_csv(index=False)))

    def test_transform_resolves_missing_features(self):
        raw = self.data.copy()
        raw.loc[0, "Age"] = None
        raw.loc[1, "Embarked"] = None
        processed = transform_titanic(raw)
        self.assertEqual(int(processed.isna().sum().sum()), 0)
        self.assertEqual(len(processed), len(raw))

    def test_uploaded_etl_creates_raw_and_processed_artifacts(self):
        root = Path(__file__).parent / ".test-data" / uuid4().hex
        raw_path = root / "raw" / "titanic_raw.csv"
        processed_path = root / "processed" / "titanic_processed.csv"
        try:
            result = run_uploaded_etl(self.data.to_csv(index=False).encode(), raw_path, processed_path)
            self.assertTrue(raw_path.exists())
            self.assertTrue(processed_path.exists())
            self.assertEqual(result.missing_after, 0)
        finally:
            for path in sorted(root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            root.rmdir()
            if root.parent.exists() and not any(root.parent.iterdir()):
                root.parent.rmdir()

    def test_accepts_collapsed_kaggle_demo_and_preserves_unique_passengers(self):
        proxy = pd.concat([self.data] * 11, ignore_index=True).head(418)
        proxy.insert(0, "PassengerId", range(892, 1310))
        lines = proxy.to_csv(index=False).splitlines()
        collapsed_rows = [f'"{line.replace(chr(34), chr(34) * 2)}"' for line in lines[1:]]
        collapsed_csv = (lines[0] + "\n" + "\n".join(collapsed_rows)).encode()
        root = Path(__file__).parent / ".test-data" / uuid4().hex
        raw_path = root / "raw" / "titanic_raw.csv"
        processed_path = root / "processed" / "titanic_processed.csv"
        try:
            result = run_uploaded_etl(collapsed_csv, raw_path, processed_path)
            self.assertEqual(result.raw_rows, 418)
            self.assertEqual(result.processed_rows, 418)
            self.assertEqual(result.source_kind, "kaggle_test_with_submission_labels")
            self.assertTrue(result.warnings)
        finally:
            for path in sorted(root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            root.rmdir()
            if root.parent.exists() and not any(root.parent.iterdir()):
                root.parent.rmdir()

    def test_trains_logistic_regression(self):
        result = train_classifier(self.data, algorithm="Logistic Regression")
        self.assertGreaterEqual(result.accuracy, 0)
        self.assertEqual(result.train_rows + result.test_rows, len(self.data))

    def test_trains_decision_tree(self):
        result = train_classifier(self.data, algorithm="Decision Tree")
        self.assertLessEqual(result.accuracy, 1)
        self.assertEqual(len(result.confusion), 2)

    def test_predicts_a_passenger(self):
        result = train_classifier(self.data)
        prediction, probability = predict_passenger(result, {"Pclass": 1, "Sex": "female", "Age": 28, "SibSp": 0, "Parch": 0, "Fare": 80.0, "Embarked": "C"})
        self.assertIn(prediction, {0, 1})
        self.assertTrue(0 <= probability <= 1)


if __name__ == "__main__":
    unittest.main()
