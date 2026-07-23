# DecodePath

DecodePath is a transparent **Tech Stack Recommender** built for DecodeLabs
Artificial Intelligence Project 3. A user supplies at least three skills plus
optional career preferences; the system converts that profile and a local
career catalog into TF-IDF vectors, measures cosine similarity, and returns
three ranked, explainable paths.

The interface is inspired by the calm conversational structure of Gemini
(light canvas, sidebar, central greeting and prompt composer) while retaining
an original DecodePath identity and a fully local deterministic engine.

## What makes it a recommendation system

```text
skills + goal + interests
          │
          ▼
canonical profile vocabulary
          │
          ▼
TF-IDF transform ───── validated career metadata
          │
          ▼
cosine similarity against all 18 roles
          │
          ▼
descending deterministic ranking → Top 3 + evidence
```

- **Content-based filtering:** the profile is compared with role content, not
  with other users.
- **TF-IDF:** distinctive terms have more influence than generic catalog terms.
- **Cosine similarity:** orientation matters more than input length.
- **Explainability:** matched skills, skill gaps, tools and learning steps are
  visible for every result.
- **Cold start:** onboarding presets and a clearly labeled popularity fallback
  operate before enough personal signal exists.
- **Persistence:** successful recommendations are stored in a local SQLite
  history and can be exported as JSON.

See [the requirement traceability](docs/project-3-requirements.md) for the
page-to-code mapping from the supplied project blueprint.

## Run the interface

From `03-decodepath-recommender`:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. All catalog data and recommendation
processing stay on the device.

## Run from the terminal

```powershell
python scripts/run_recommender.py Python SQL Statistics "Machine Learning" `
  --goal "Move into an AI role"
```

The command emits the same Top 3 contract as formatted JSON.

## Test

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
python -m unittest discover -s tests -v
```

The suite checks catalog integrity, free-text aliases, the three-skill
threshold, relevance, bounded scores, deterministic ranking, cold start and
SQLite history.

## Project structure

```text
03-decodepath-recommender/
├── app.py
├── data/raw/raw_skills.csv
├── scripts/run_recommender.py
├── src/
│   ├── catalog.py
│   ├── models.py
│   ├── recommender.py
│   └── storage.py
├── tests/test_recommender.py
└── docs/project-3-requirements.md
```

## Scope

This is an educational, deterministic recommendation system. Similarity is not
a job-success probability, and the curated catalog is not a live labor-market
forecast.

## Author

Jean Franck Loa Rojas
