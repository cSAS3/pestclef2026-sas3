# PestCLEF 2026 SAS3 reproducibility code

This repository contains public reproducibility materials for Team SAS3 / Seine A. Shintani's PestCLEF 2026 working note:

**Hybrid Retrieval-Guided Knowledge Graph Extraction with Precision Repair for PestCLEF 2026**

Final selected runs:

- Public F-score: 0.45603
- Private F-score: 0.50952
- Final private leaderboard rank: 1st

## Contents

```text
src/           Candidate generation, candidate ranking, final-patch application, comparison, and validation scripts.
final_runs/    Final selected submission CSV files.
patches/       Explicit patch specification for the final selected runs.
reports/       Validation reports and construction summaries.
data/          Placeholder only; official raw data are not redistributed.
```

## What is included

The repository includes:

- candidate-generation and candidate-ranking scripts;
- final selected run CSV files;
- explicit final-run patch specifications;
- validation scripts;
- validation reports;
- reproducibility notes.

## What is not included

This repository intentionally does **not** include:

- official PestCLEF/EPOP raw documents;
- train/development/test JSON files;
- EPOP document text;
- private competition data;
- broad internal experiment logs;
- public-relations or administrative materials;
- manuscript PDFs or unpublished paper source files.

The official data should be obtained from the authorized CLEF/LifeCLEF/PestCLEF channels and used under the task data-use terms.

## Quick validation

Validate the final selected runs:

```bash
python src/validate_submission.py final_runs/PestCLEF2026_RunA_MonopoliOnly.csv
python src/validate_submission.py final_runs/PestCLEF2026_RunB_MonopoliPlusPolignano.csv
```

Compare the two final selected runs:

```bash
python src/compare_submissions.py final_runs/PestCLEF2026_RunA_MonopoliOnly.csv final_runs/PestCLEF2026_RunB_MonopoliPlusPolignano.csv
```

## Re-running the candidate pipeline

A full rerun requires the official PestCLEF train/development/test JSON files and EPOP document texts. Place local copies under `data/` as described in `data/README.md`. These official files are not redistributed here.

## Citation

Please cite the accompanying PestCLEF 2026 working note and this repository. A `CITATION.cff` file is provided.
