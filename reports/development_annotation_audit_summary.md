# Development annotation audit summary

This report summarizes the development-set audit used in the PestCLEF 2026 SAS3 system. It does not include the official raw PestCLEF data.

Key observations:

- The submission target is a document-level knowledge graph.
- Empty or low-edge documents make precision control important.
- `Located_in` was the highest-risk and highest-frequency predicate during development.
- Alias and normalization variation were important sources of recall errors.
- The system therefore used schema constraints, alias matching, duplicate-aware retrieval, candidate verification, and final precision repair.

The detailed raw annotations and EPOP document texts are not redistributed in this repository. Users should obtain the official data from the PestCLEF/LifeCLEF channels and place them under `data/` as described in `data/README.md`.
