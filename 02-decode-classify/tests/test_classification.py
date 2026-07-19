import unittest
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.data import FEATURES, SPECIES, TARGET, dataset_summary, load_dataset
from src.etl import run_etl, run_uploaded_etl
from src.etl.extract import extract_iris
from src.etl.transform import transform_iris
from src.training import predict_flower, train_classifier, tune_neighbors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATASET = PROJECT_ROOT / "data" / "raw" / "iris.csv"


def iris_data() -> pd.DataFrame:
    return transform_iris(extract_iris(RAW_DATASET))


class ClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = iris_data()

    def test_loads_and_summarizes_official_iris_dataset(self):
        summary = dataset_summary(self.data)
        self.assertEqual(summary["rows"], 150)
        self.assertEqual(summary["features"], 4)
        self.assertEqual(summary["classes"], 3)
        self.assertEqual(summary["missing"], 0)
        self.assertEqual(summary["class_counts"], {species: 50 for species in SPECIES})

    def test_load_dataset_enforces_the_canonical_contract(self):
        loaded = load_dataset(StringIO(self.data.to_csv(index=False)))
        self.assertEqual(tuple(loaded.columns), (*FEATURES, TARGET))
        self.assertTrue(all(pd.api.types.is_float_dtype(loaded[column]) for column in FEATURES))

    def test_rejects_missing_columns(self):
        invalid = self.data.drop(columns=[FEATURES[-1]])
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            extract_iris(StringIO(invalid.to_csv(index=False)))

    def test_rejects_unknown_species(self):
        invalid = self.data.copy()
        invalid.loc[0, TARGET] = "unknown"
        with self.assertRaisesRegex(ValueError, "unsupported labels"):
            transform_iris(invalid)

    def test_rejects_non_positive_measurements(self):
        invalid = self.data.copy()
        invalid.loc[0, FEATURES[0]] = 0
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            transform_iris(invalid)

    def test_transform_removes_duplicate_model_rows(self):
        duplicated = pd.concat(
            [self.data, self.data.iloc[[0]]],
            ignore_index=True,
        )
        processed = transform_iris(duplicated)
        self.assertEqual(len(processed), 150)
        self.assertFalse(processed.duplicated().any())

    def test_accepts_common_sklearn_column_names_and_numeric_targets(self):
        aliases = self.data.rename(
            columns={
                FEATURES[0]: "sepal length (cm)",
                FEATURES[1]: "sepal width (cm)",
                FEATURES[2]: "petal length (cm)",
                FEATURES[3]: "petal width (cm)",
                TARGET: "target",
            }
        )
        aliases["target"] = aliases["target"].map(dict(zip(SPECIES, range(3))))
        extracted = extract_iris(StringIO(aliases.to_csv(index=False)))
        transformed = transform_iris(extracted)
        self.assertEqual(tuple(transformed.columns), (*FEATURES, TARGET))
        self.assertEqual(set(transformed[TARGET]), set(SPECIES))

    def test_run_etl_creates_a_canonical_processed_artifact(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "iris_processed.csv"
            result = run_etl(RAW_DATASET, output)
            self.assertTrue(output.exists())
            self.assertEqual(result.raw_rows, 150)
            self.assertEqual(result.processed_rows, 150)
            self.assertEqual(result.source_kind, "sklearn_iris")
            self.assertEqual(result.missing_after, 0)

    def test_uploaded_etl_preserves_valid_raw_and_processed_artifacts(self):
        upload = self.data.to_csv(index=False).encode("utf-8")
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_path = root / "raw" / "iris.csv"
            processed_path = root / "processed" / "iris_processed.csv"
            result = run_uploaded_etl(upload, raw_path, processed_path)
            self.assertEqual(raw_path.read_bytes(), upload)
            self.assertTrue(processed_path.exists())
            self.assertEqual(result.class_counts, {species: 50 for species in SPECIES})

    def test_trains_knn_with_reproducible_multiclass_metrics(self):
        first = train_classifier(self.data, neighbors=5)
        second = train_classifier(self.data, neighbors=5)
        self.assertGreaterEqual(first.accuracy, 0.85)
        self.assertEqual(first.accuracy, second.accuracy)
        self.assertEqual(first.macro_f1, second.macro_f1)
        self.assertEqual(first.train_rows + first.test_rows, len(self.data))
        self.assertEqual(first.class_names, SPECIES)
        self.assertEqual(len(first.confusion), 3)
        self.assertEqual(sum(map(sum, first.confusion)), first.test_rows)

    def test_rejects_neighbors_larger_than_the_training_partition(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            train_classifier(self.data, neighbors=149)

    def test_predicts_a_flower_with_normalized_probabilities(self):
        result = train_classifier(self.data, neighbors=5)
        prediction, confidence, probabilities = predict_flower(
            result,
            {
                "sepal_length_cm": 5.1,
                "sepal_width_cm": 3.5,
                "petal_length_cm": 1.4,
                "petal_width_cm": 0.2,
            },
        )
        self.assertIn(prediction, SPECIES)
        self.assertEqual(set(probabilities), set(SPECIES))
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        self.assertAlmostEqual(confidence, probabilities[prediction])

    def test_tunes_neighbor_values_with_stratified_cross_validation(self):
        tuning = tune_neighbors(self.data, max_neighbors=7, folds=5)
        self.assertEqual(
            list(tuning.columns),
            ["neighbors", "mean_accuracy", "std_accuracy"],
        )
        self.assertEqual(tuning["neighbors"].tolist(), list(range(1, 8)))
        self.assertTrue(tuning["mean_accuracy"].between(0, 1).all())
        self.assertTrue(tuning["std_accuracy"].ge(0).all())

    def test_tuning_rejects_an_invalid_holdout_size(self):
        with self.assertRaisesRegex(ValueError, "test_size"):
            tune_neighbors(self.data, test_size=1.0)


if __name__ == "__main__":
    unittest.main()
