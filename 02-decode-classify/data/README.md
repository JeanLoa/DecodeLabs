# Data workspace

- `raw/titanic_raw.csv` preserves the source exactly as received.
- `processed/titanic_processed.csv` contains the validated, cleaned model-ready dataset.

The bundled raw demo source is versioned so the application can initialize
automatically after a clone. Processed CSV and quality-report artifacts are
reproducible ETL outputs and remain excluded from Git.
