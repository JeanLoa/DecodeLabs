# DecodeClassify

DecodeClassify is an interactive supervised-learning lab for DecodeLabs
Artificial Intelligence Project 2. It follows the project blueprint directly:
load the Iris dataset, separate training and validation data, standardize the
four measurements, train a K-Nearest Neighbors classifier, and inspect more
than accuracy before classifying a new flower.

## Requirement coverage

| DecodeLabs blueprint | Implementation |
|---|---|
| Small, understandable dataset | 150 Iris samples, 3 balanced species, 4 measurements |
| Input / process / output workflow | Validated CSV → split and scale → prediction and metrics |
| Randomized train-test split | Reproducible, shuffled, stratified 80/20 split by default |
| Feature scaling | `StandardScaler` fitted only on training data inside the model pipeline |
| Simple classifier | `KNeighborsClassifier`, with an interactive odd `K` value |
| Tune `K` | Stratified cross-validation curve that does not inspect the held-out test set |
| Validate output | Accuracy, macro F1, per-class precision/recall/F1, a 3×3 confusion matrix, and one-vs-rest TP/FP/FN/TN |
| Test new data | Interactive four-measurement flower classifier with class probabilities |

See the [page-by-page requirement traceability](docs/project-2-requirements.md)
for the complete mapping from the PDF to repository evidence.

## Architecture

```text
data/raw/iris.csv
        │  extract + schema validation
        ▼
src/etl/transform.py
        │  normalize types, labels, and duplicates
        ▼
data/processed/iris_processed.csv
        │
        ├── stratified train set ── StandardScaler.fit ── KNN.fit
        │
        └── held-out test set  ── StandardScaler.transform ── evaluate
```

Scaling lives in the scikit-learn `Pipeline`, after the split. This prevents
information from the validation set leaking into the values learned by the
scaler.

## Run

From `02-decode-classify`:

```powershell
python -m pip install -r requirements.txt
python scripts/run_etl.py
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. The app includes four workspaces:

- **Overview** — understand class balance, feature ranges, and the full pipeline.
- **Train model** — choose `K`, run the split/training flow, and inspect metrics.
- **Predict** — enter a flower's four measurements and review probabilities.
- **Dataset** — inspect raw/processed rows, schema, and a scaling demonstration.

Prediction remains locked until an experiment has been run in the current
session, so every result is tied to a visible `K` and test-size configuration.

## Test

```powershell
python -m unittest discover -s tests -v
```

The tests cover dataset validation, ETL behavior, split integrity, scaling,
KNN training, multiclass evaluation, neighbor tuning, and prediction.

If a Windows/Conda environment over-allocates numerical-library threads, start
a fresh PowerShell and constrain them for this small dataset:

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
python -m unittest discover -s tests -v
```

## Project structure

```text
02-decode-classify/
├── app.py
├── data/
│   ├── raw/iris.csv
│   └── processed/                 # reproducible, ignored artifacts
├── scripts/run_etl.py
├── src/
│   ├── data.py
│   ├── training.py
│   └── etl/
└── tests/test_classification.py
```

## Reproducibility and scope

- `random_state=42` is used for the shuffled split and cross-validation.
- The held-out test set is used once for final evaluation, not to select `K`.
- The bundled CSV keeps the app fully local and reproducible.
- This is a small educational benchmark, not a production or botanical
  decision system.

## Author

Jean Franck Loa Rojas
