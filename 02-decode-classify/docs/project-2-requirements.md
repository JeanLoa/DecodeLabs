# DecodeLabs Project 2 — Requirement traceability

This checklist maps the 21-page Project 2 brief to concrete repository
evidence. The PDF describes the conceptual architecture; it does not prescribe
a UI, folder structure, deployment target, or minimum score.

| PDF evidence | Requirement | Repository evidence | Status |
|---|---|---|---|
| Page 4 | Load and understand a small classification dataset | `data/raw/iris.csv`, `src/data.py`, Overview and Dataset workspaces | Implemented |
| Pages 7–8 | Use Iris: 150 samples, 3 balanced classes, 4 dimensions | Canonical CSV and schema validation | Implemented |
| Pages 8, 17 | Use sepal/petal length and width to classify Setosa, Versicolor, and Virginica | `FEATURES`, `SPECIES`, prediction form | Implemented |
| Pages 7, 9 | Standardize features to mean 0 and variance 1 | `StandardScaler` inside the model `Pipeline`; scaling explanation in the app | Implemented |
| Page 10 | Shuffle before splitting to avoid order bias | Explicit `shuffle=True`, stratification, and `random_state=42` | Implemented |
| Page 17 | Use an 80/20 train-test split | `test_size=0.2` default and 20% default in the experiment UI | Implemented |
| Pages 7, 11, 13 | Train K-Nearest Neighbors, with K=5 as the reference | `KNeighborsClassifier(n_neighbors=5)` default | Implemented |
| Page 13 | Execute `fit` on training data and `predict` on test data | `train_classifier` in `src/training.py` | Implemented |
| Page 12 | Compare K values and explain overfitting/underfitting | Training-only stratified cross-validation and error-rate curve for K=1…15 | Implemented |
| Pages 7, 15, 17 | Generate a confusion matrix | Multiclass 3×3 matrix in `TrainingResult` and the Train workspace | Implemented |
| Page 15 | Explain TP, FP, FN, and TN | One-vs-rest diagnostic table for every Iris species | Implemented |
| Pages 14, 16 | Evaluate beyond accuracy | Per-class precision, recall and F1; macro and weighted F1 | Implemented |
| Pages 7, 17 | Show the complete input → process → output flow | Overview architecture, ETL modules, model pipeline, evaluation output | Implemented |
| Page 20 | Test completely new data | Interactive prediction workspace with all class probabilities | Implemented |

## Integrity decisions

- Scaling is fitted after the split and only on training data.
- K tuning first reserves the same stratified holdout, then cross-validates only
  on the remaining training partition.
- The held-out test set is evaluated after configuration and is not used to
  choose K.
- Prediction remains locked until an experiment has run in the current session.
- Raw data is versioned; processed data and ETL reports are reproducible
  generated artifacts.

## Runtime acceptance

Code and documentation are prepared for the following user-run acceptance
commands:

```powershell
python scripts/run_etl.py
python -m unittest discover -s tests -v
python -m streamlit run app.py
```

The automated suite and final Streamlit walkthrough are intentionally left for
the user to execute.
