#!/usr/bin/env python3
"""Apply the final selected PestCLEF 2026 SAS3 patch triples to a pre-final backbone.

This script is intentionally small and transparent. The full competition system contained
candidate generation, verification, and manual precision repair. The final selected runs
can be described as small patches over a pre-final validated backbone. Because the raw
PestCLEF/EPOP data and the original intermediate work directories are not redistributed,
this script is meant to document and reproduce the final patching step when the user
provides the corresponding backbone CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

PATCHES = {
    "A": [
        {"doc_id": "102433", "predicate": "Transmits", "subject": "Asian citrus psyllid", "object": "Candidatus Liberibacter asiaticus"},
        {"doc_id": "102433", "predicate": "Transmits", "subject": "Asian citrus psyllid", "object": "CLas"},
        {"doc_id": "100506", "predicate": "Located_in", "subject": "Xylella", "object": "countryside of Monopoli"},
    ],
    "B": [
        {"doc_id": "102433", "predicate": "Transmits", "subject": "Asian citrus psyllid", "object": "Candidatus Liberibacter asiaticus"},
        {"doc_id": "102433", "predicate": "Transmits", "subject": "Asian citrus psyllid", "object": "CLas"},
        {"doc_id": "100506", "predicate": "Located_in", "subject": "Xylella", "object": "countryside of Monopoli"},
        {"doc_id": "100506", "predicate": "Located_in", "subject": "Xylella", "object": "countryside of Polignano"},
    ],
}


def edge_key(edge: dict[str, str]) -> tuple[str, str, str]:
    return edge["predicate"], edge["subject"], edge["object"]


def load_submission(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if set(reader.fieldnames or []) != {"doc_id", "knowledge_graph"}:
            raise ValueError(f"Expected doc_id,knowledge_graph columns; got {reader.fieldnames}")
        for row in reader:
            rows.append({"doc_id": str(row["doc_id"]), "knowledge_graph": json.loads(row["knowledge_graph"])})
    return rows


def write_submission(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "knowledge_graph"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "doc_id": row["doc_id"],
                "knowledge_graph": json.dumps(row["knowledge_graph"], ensure_ascii=False),
            })


def apply_patch(rows: list[dict[str, object]], patch: list[dict[str, str]]) -> list[dict[str, object]]:
    by_doc = {str(row["doc_id"]): list(row["knowledge_graph"]) for row in rows}
    for item in patch:
        doc_id = item["doc_id"]
        if doc_id not in by_doc:
            raise KeyError(f"Patch refers to document {doc_id}, which is absent from the input submission")
        edge = {k: item[k] for k in ["predicate", "subject", "object"]}
        seen = {edge_key(e) for e in by_doc[doc_id]}
        if edge_key(edge) not in seen:
            by_doc[doc_id].append(edge)
    return [{"doc_id": row["doc_id"], "knowledge_graph": by_doc[str(row["doc_id"])]} for row in rows]


def validate(rows: list[dict[str, object]]) -> dict[str, object]:
    report = {"rows": len(rows), "total_triples": 0, "empty_documents": 0, "duplicate_triples": 0, "bad_edges": []}
    doc_ids = set()
    for row in rows:
        doc_id = str(row["doc_id"])
        if doc_id in doc_ids:
            raise ValueError(f"duplicate doc_id: {doc_id}")
        doc_ids.add(doc_id)
        kg = row["knowledge_graph"]
        if not isinstance(kg, list):
            raise ValueError(f"knowledge_graph is not a list for doc_id={doc_id}")
        if not kg:
            report["empty_documents"] += 1
        seen = set()
        for edge in kg:
            if not isinstance(edge, dict) or set(edge) != {"predicate", "subject", "object"}:
                report["bad_edges"].append({"doc_id": doc_id, "edge": edge})
                continue
            if not all(isinstance(edge[k], str) for k in ["predicate", "subject", "object"]):
                report["bad_edges"].append({"doc_id": doc_id, "edge": edge})
                continue
            key = edge_key(edge)
            if key in seen:
                report["duplicate_triples"] += 1
            seen.add(key)
            report["total_triples"] += 1
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="Pre-final backbone submission CSV")
    parser.add_argument("--run", choices=["A", "B"], required=True, help="Final run variant to create")
    parser.add_argument("--output", required=True, type=Path, help="Output submission CSV")
    parser.add_argument("--report", type=Path, help="Optional validation report JSON")
    args = parser.parse_args()

    rows = load_submission(args.input)
    patched = apply_patch(rows, PATCHES[args.run])
    report = validate(patched)
    write_submission(patched, args.output)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
