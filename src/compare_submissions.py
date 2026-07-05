#!/usr/bin/env python3
"""Compare two PestCLEF submission CSV files and list triple differences."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path


def read(path: Path) -> dict[str, set[tuple[str,str,str]]]:
    data={}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            data[str(row["doc_id"])] = {(e["predicate"], e["subject"], e["object"]) for e in json.loads(row["knowledge_graph"])}
    return data

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("left", type=Path)
    ap.add_argument("right", type=Path)
    ap.add_argument("--csv", type=Path)
    args=ap.parse_args()
    A, B = read(args.left), read(args.right)
    rows=[]
    for doc_id in sorted(set(A) | set(B)):
        for pred,sub,obj in sorted(B.get(doc_id,set()) - A.get(doc_id,set())):
            rows.append({"doc_id": doc_id, "side": "right_only", "predicate": pred, "subject": sub, "object": obj})
        for pred,sub,obj in sorted(A.get(doc_id,set()) - B.get(doc_id,set())):
            rows.append({"doc_id": doc_id, "side": "left_only", "predicate": pred, "subject": sub, "object": obj})
    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer=csv.DictWriter(f, fieldnames=["doc_id", "side", "predicate", "subject", "object"])
            writer.writeheader(); writer.writerows(rows)
    for r in rows:
        print(f"{r['doc_id']}\t{r['side']}\t{r['predicate']}\t{r['subject']}\t{r['object']}")
