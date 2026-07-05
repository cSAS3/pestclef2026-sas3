#!/usr/bin/env python3
"""Validate a PestCLEF submission CSV."""
from __future__ import annotations
import argparse, csv, json
from collections import Counter
from pathlib import Path

PREDICATES = {"Located_in", "Found_on", "Occurs_on", "Causes", "Affects", "Dispersed_by", "Transmits"}

def validate(path: Path) -> dict:
    report = {
        "file": str(path), "rows": 0, "total_triples": 0, "empty_documents": 0,
        "json_errors": [], "bad_edges": [], "duplicate_doc_ids": [], "duplicate_triples": 0,
        "predicate_counts": {},
    }
    doc_ids=set(); pred=Counter()
    with path.open(newline="", encoding="utf-8") as f:
        reader=csv.DictReader(f)
        if set(reader.fieldnames or []) != {"doc_id", "knowledge_graph"}:
            raise ValueError(f"Expected doc_id,knowledge_graph columns; got {reader.fieldnames}")
        for row in reader:
            report["rows"] += 1
            doc_id=str(row["doc_id"])
            if doc_id in doc_ids: report["duplicate_doc_ids"].append(doc_id)
            doc_ids.add(doc_id)
            try:
                kg=json.loads(row["knowledge_graph"])
            except Exception as exc:
                report["json_errors"].append({"doc_id": doc_id, "error": str(exc)})
                continue
            if not kg: report["empty_documents"] += 1
            seen=set()
            for edge in kg:
                if not isinstance(edge, dict) or set(edge) != {"predicate", "subject", "object"}:
                    report["bad_edges"].append({"doc_id": doc_id, "edge": edge}); continue
                if edge["predicate"] not in PREDICATES:
                    report["bad_edges"].append({"doc_id": doc_id, "edge": edge, "reason": "unknown predicate"}); continue
                if not all(isinstance(edge[k], str) for k in ["predicate", "subject", "object"]):
                    report["bad_edges"].append({"doc_id": doc_id, "edge": edge, "reason": "non-string field"}); continue
                key=(edge["predicate"], edge["subject"], edge["object"])
                if key in seen: report["duplicate_triples"] += 1
                seen.add(key)
                pred[edge["predicate"]] += 1
                report["total_triples"] += 1
    report["predicate_counts"] = dict(pred)
    return report

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--json", type=Path)
    args=ap.parse_args()
    rep=validate(args.csv)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    if args.json:
        args.json.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
