# DecodeClassify

DecodeClassify is a GitHub Copilot-inspired classification workspace built for
DecodeLabs Artificial Intelligence Project 2. It trains and evaluates a simple
supervised-learning model using the Titanic dataset published on Kaggle.

## Data architecture

```text
data/raw/titanic_raw.csv
        ↓ extract + validate
src/etl/transform.py
        ↓ normalize + impute + enforce contract
data/processed/titanic_processed.csv
        ↓
src/training.py
```

The raw source is preserved unchanged. Training uses only the processed artifact.
The bundled 418-row demo combines Kaggle test passengers with submission-derived
labels so the complete local workflow can run. The interface marks this provenance
because those labels are not verified ground truth and its metrics are demonstrative.
On startup, the application reads the versioned raw source and regenerates the
processed artifact automatically.

Run ETL explicitly with:

```powershell
python scripts/run_etl.py
```

The command writes `data/processed/etl_report.json` with row counts, missing-value
quality, provenance, warnings and completion or rejection status.

## Features

- Validated CSV loading and dataset overview.
- Reproducible stratified train/test split.
- Logistic Regression and Decision Tree classifiers.
- Accuracy, confusion matrix, precision, recall and F1 score.
- Interactive passenger prediction with model confidence.
- Local-only processing and explicit educational limitations.
- Automated tests for validation, training and prediction.

## Dataset

Download `train.csv` from the Apache 2.0 Kaggle dataset:

https://www.kaggle.com/competitions/titanic/data?select=train.csv

To use official training ground truth, place its contents at:

```text
02-decode-classify/data/raw/titanic_raw.csv
```

Alternatively, upload `train.csv` from the Streamlit sidebar. The application
does not redistribute the dataset or require Kaggle credentials at runtime.

## Run

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Test

```powershell
python -m unittest discover -s tests -v
```

## Project requirement coverage

| Requirement | Implementation |
|---|---|
| Load and understand a dataset | Validated loader, summary and explorer |
| Split training and testing data | Stratified `train_test_split` |
| Apply a classification algorithm | Logistic Regression and Decision Tree |
| Practice supervised learning | Training, evaluation and new predictions |

## Scope

This is an educational binary-classification project. It does not claim that
historical survival outcomes represent modern safety, fairness or causal rules.

## Author

Jean Franck Loa Rojas
