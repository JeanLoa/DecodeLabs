# Data workspace

- `raw/iris.csv` is the versioned canonical input generated from
  `sklearn.datasets.load_iris`.
- `processed/iris_processed.csv` is the validated, canonical ETL output used by
  model training.
- `processed/etl_report.json` records ETL status, quality counts, provenance and
  class balance.

The raw dataset is never edited in place by the regular ETL command. Processed
CSV and quality-report artifacts are reproducible outputs and remain excluded
from Git.
