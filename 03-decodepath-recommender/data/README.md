# Local recommendation data

`raw/raw_skills.csv` is the versioned content catalog used by DecodePath. Each
row represents a recommendable career path and contains:

- explicit skill and tool metadata;
- search and career keywords;
- a four-step learning path;
- a bounded popularity score used only for the cold-start fallback.

The application does not download or silently mutate this dataset. Runtime
history is stored in `data/decodepath.db`, which is intentionally ignored by
Git.
