# Project 3 requirement traceability

This document maps the DecodeLabs **AI Recommendation Logic** blueprint in
`03-project.pdf` to concrete implementation evidence.

| Blueprint requirement | Implementation evidence |
|---|---|
| Accept user choices or interests | `app.py` captures at least three explicit skills, a goal, free-form interests and experience; it also parses a natural-language prompt. |
| Recommend a technology path | `data/raw/raw_skills.csv` contains 18 career paths with skill, tool, keyword and learning-path metadata. |
| Use content-based filtering | `src/recommender.py` compares one user profile with item metadata; it never depends on other users. |
| Shared user/item vocabulary | Catalog labels and `SKILL_ALIASES` resolve free text to canonical technology terms before vectorization. |
| Prefer TF-IDF to binary overlap | `TfidfVectorizer` weights distinctive unigram/bigram terms and uses sublinear term frequency. |
| Use cosine similarity | `cosine_similarity` compares the profile vector with every career vector and exposes the bounded raw score. |
| Ingestion → scoring → sorting → Top-N | Catalog validation, vector scoring, deterministic descending sort and a default Top 3 are separate pipeline stages. |
| Minimum three user inputs | `MINIMUM_SKILLS = 3` is enforced by the engine, UI and tests. |
| Show the three most relevant careers | Every successful interaction returns exactly three explainable results. |
| Address user cold start | Onboarding presets gather explicit preferences; the method view clearly presents a popularity fallback before personalization. |
| Address item cold start | Catalog validation rejects items with fewer than three skills or missing tool, keyword and learning-path metadata. |
| Explain and improve results | Each result shows overlap, skill gaps, useful tools, a learning path and a score caveat; JSON export supports later analysis. |
| Verify quality | `tests/test_recommender.py` covers validation, aliases, input thresholds, relevance, score bounds, deterministic ordering, cold start and persistence. |

## Honest scope boundary

DecodePath is a deterministic educational recommender, not a labor-market
forecast or hiring decision system. Its results are limited by the curated
local catalog and the preferences a user explicitly provides.
