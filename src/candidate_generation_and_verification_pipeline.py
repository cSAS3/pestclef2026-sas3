#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import io
import json
import math
import os
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import regex
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold

ARG_MAP = {
    "Affects": ("Disease", "Plant"),
    "Causes": ("Pest", "Disease"),
    "Dispersed_by": ("Nuisance", "Dissemination_pathway"),
    "Found_on": ("Organism", "Habitat"),
    "Located_in": ("Object", "Location"),
    "Occurs_on": ("Object", "Date"),
    "Transmits": ("Vector", "Nuisance"),
}

SCHEMA = {
    "Affects": ({"Disease"}, {"Plant"}),
    "Causes": ({"Pest"}, {"Disease"}),
    "Dispersed_by": ({"Disease", "Pest"}, {"Dissemination_pathway"}),
    "Found_on": ({"Pest", "Vector"}, {"Plant", "Dissemination_pathway"}),
    "Located_in": ({"Disease", "Pest", "Plant", "Vector", "Dissemination_pathway"}, {"Location"}),
    "Occurs_on": ({"Disease", "Pest", "Plant", "Vector", "Dissemination_pathway"}, {"Date"}),
    "Transmits": ({"Vector"}, {"Disease", "Pest"}),
}

PREDICATES = ["Located_in", "Found_on", "Occurs_on", "Causes", "Affects", "Dispersed_by", "Transmits"]


@dataclass
class Cluster:
    concept_key: str
    type: str
    aliases: Tuple[str, ...]
    norms: Tuple[Tuple[str, str], ...]
    representative: str


@dataclass
class DocInfo:
    doc_id: str
    split: str
    text: str
    lang: str
    url: str
    title: str
    lead: str
    layout: list
    title_spans: List[Tuple[int, int]]
    lead_spans: List[Tuple[int, int]]
    paragraphs: List[Tuple[int, int]]
    sent_spans: List[Tuple[int, int]]
    clusters: Dict[str, Cluster]
    cluster_by_concept: Dict[str, List[str]]
    gold_edges: set[Tuple[str, str, str]]


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_alias(alias: str) -> str:
    alias = html.unescape(alias)
    alias = alias.replace("\\/", "/")
    alias = re.sub(r"<[^>]+>", "", alias)
    alias = re.sub(r"\s+", " ", alias).strip()
    return alias


def normalize_surface(text: str) -> str:
    return re.sub(r"\s+", " ", clean_alias(text)).strip().casefold()


def norm_tuple(entity: dict) -> Tuple[Tuple[str, str], ...]:
    norms = []
    for item in entity.get("normalizations", []):
        resource = item.get("resource")
        reference = item.get("reference")
        if resource and reference:
            norms.append((str(resource), str(reference)))
    return tuple(sorted(set(norms)))


def make_concept_key(entity_type: str, norms: Tuple[Tuple[str, str], ...], aliases: Iterable[str]) -> str:
    if norms:
        return entity_type + "|" + "|".join(f"{resource}:{reference}" for resource, reference in norms)
    surfaces = sorted({normalize_surface(alias) for alias in aliases if normalize_surface(alias)}, key=lambda s: (-len(s), s))
    return f"{entity_type}|surface:{surfaces[0] if surfaces else ''}"


def build_union_find(doc: dict):
    entities = {entity["id"]: entity for entity in doc.get("text_bound_annotations", {}).get("entities", [])}
    parent = {eid: eid for eid in entities}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for group in doc.get("text_bound_annotations", {}).get("identity_coreferences", []):
        for a, b in zip(group, group[1:]):
            union(a, b)

    buckets: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], List[str]] = defaultdict(list)
    for eid, entity in entities.items():
        norms = norm_tuple(entity)
        if norms:
            buckets[(entity["type"], norms)].append(eid)
    for ids in buckets.values():
        for a, b in zip(ids, ids[1:]):
            union(a, b)

    clusters: Dict[str, List[str]] = defaultdict(list)
    for eid in entities:
        clusters[find(eid)].append(eid)
    return entities, clusters, find


def sentence_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"(?<=[\.\!\?\n])\s+", text):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(text)))
    return spans


def extract_sections(text: str, layout: list[dict]) -> tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]], str, str]:
    items = sorted(layout, key=lambda x: (x["offsets"][0], x["offsets"][1], x["tag"]))
    title_spans = [tuple(item["offsets"]) for item in items if item["tag"] == "h1"]
    paragraphs = [tuple(item["offsets"]) for item in items if item["tag"] in {"p", "li", "h1", "h2", "h3", "h4", "h5"}]
    lead_spans: List[Tuple[int, int]] = []
    first_title_end = title_spans[0][1] if title_spans else 0
    count = 0
    for item in items:
        if item["tag"] in {"p", "li"} and item["offsets"][0] >= first_title_end:
            lead_spans.append(tuple(item["offsets"]))
            count += 1
            if count >= 2:
                break
    title = " ".join(text[s:e] for s, e in title_spans).strip()
    lead = " ".join(text[s:e] for s, e in lead_spans).strip()
    return title_spans, lead_spans, paragraphs, title, lead


def load_texts_and_metadata(zip_path: str | Path) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, dict[str, str]]]]:
    texts: dict[str, dict[str, str]] = {"train": {}, "dev": {}, "test": {}}
    metadata: dict[str, dict[str, dict[str, str]]] = {"train": {}, "dev": {}, "test": {}}
    with zipfile.ZipFile(zip_path) as zf:
        for split in ["train", "dev", "test"]:
            prefix = f"EPOP_documents/{split}/"
            for name in zf.namelist():
                if name.startswith(prefix) and name.endswith(".txt"):
                    texts[split][Path(name).stem] = zf.read(name).decode("utf-8")
            meta_name = f"EPOP_documents/{split}/documents-metadata.csv"
            meta_df = pd.read_csv(io.BytesIO(zf.read(meta_name)), sep="\t")
            meta_df["DOC"] = meta_df["DOC"].astype(str)
            metadata[split] = meta_df.set_index("DOC").to_dict("index")
    return texts, metadata


def parse_doc(doc: dict, split: str, texts: dict[str, dict[str, str]], metadata: dict[str, dict[str, dict[str, str]]]) -> DocInfo:
    doc_id = str(doc["doc_id"])
    text = texts[split][doc_id]
    title_spans, lead_spans, paragraphs, title, lead = extract_sections(text, doc["layout"])
    entities, clusters, find = build_union_find(doc)

    cluster_objs: Dict[str, Cluster] = {}
    cluster_by_concept: Dict[str, List[str]] = defaultdict(list)
    for root, ids in clusters.items():
        member_entities = [entities[eid] for eid in ids]
        entity_type = member_entities[0]["type"]
        aliases = tuple(sorted({clean_alias(entity["form"]) for entity in member_entities}))
        norms = tuple(sorted({norm for entity in member_entities for norm in norm_tuple(entity)}))
        concept_key = make_concept_key(entity_type, norms, aliases)
        representative = sorted(aliases, key=lambda s: (-len(s), s))[0] if aliases else ""
        cluster_objs[root] = Cluster(concept_key=concept_key, type=entity_type, aliases=aliases, norms=norms, representative=representative)
        cluster_by_concept[concept_key].append(root)

    gold_edges: set[Tuple[str, str, str]] = set()
    for rel in doc.get("text_bound_annotations", {}).get("relations", []):
        subj_arg, obj_arg = ARG_MAP[rel["type"]]
        sroot = find(rel["args"][subj_arg])
        oroot = find(rel["args"][obj_arg])
        gold_edges.add((rel["type"], cluster_objs[sroot].concept_key, cluster_objs[oroot].concept_key))

    meta = metadata[split].get(doc_id, {})
    return DocInfo(
        doc_id=doc_id,
        split=split,
        text=text,
        lang=str(meta.get("LANG", "")),
        url=str(meta.get("URL", "")),
        title=title,
        lead=lead,
        layout=doc["layout"],
        title_spans=title_spans,
        lead_spans=lead_spans,
        paragraphs=paragraphs,
        sent_spans=sentence_spans(text),
        clusters=cluster_objs,
        cluster_by_concept=dict(cluster_by_concept),
        gold_edges=gold_edges,
    )


def build_test_stub(doc: dict) -> dict:
    return {
        "doc_id": doc["doc_id"],
        "layout": doc["layout"],
        "text_bound_annotations": {"entities": [], "relations": [], "identity_coreferences": []},
    }


def alias_pattern(alias: str) -> regex.Pattern:
    escaped = regex.escape(alias)
    start = r"(?<![\p{L}\p{N}])" if alias and alias[0].isalnum() else ""
    end = r"(?![\p{L}\p{N}])" if alias and alias[-1].isalnum() else ""
    return regex.compile(start + escaped + end, flags=regex.IGNORECASE)


class MajorRevamp:
    def __init__(self, train_docs: list[dict], dev_docs: list[dict], test_docs: list[dict], texts_zip: str | Path, reference_csv: Optional[str] = None):
        texts, metadata = load_texts_and_metadata(texts_zip)
        self.labeled_docs: dict[str, DocInfo] = {}
        self.test_docs: dict[str, DocInfo] = {}
        for doc in train_docs:
            info = parse_doc(doc, "train", texts, metadata)
            self.labeled_docs[info.doc_id] = info
        for doc in dev_docs:
            info = parse_doc(doc, "dev", texts, metadata)
            self.labeled_docs[info.doc_id] = info
        for doc in test_docs:
            info = parse_doc(build_test_stub(doc), "test", texts, metadata)
            self.test_docs[info.doc_id] = info

        self.all_docs: dict[str, DocInfo] = {**self.labeled_docs, **self.test_docs}
        self.labeled_ids = list(self.labeled_docs)
        self.test_ids = list(self.test_docs)
        self.reference_csv = reference_csv

        self.alias_entries = self.build_alias_entries()
        self.matched_mentions = {doc_id: self.match_doc(doc_id) for doc_id in self.all_docs}
        self.profiles = {doc_id: self.build_profile(doc_id) for doc_id in self.all_docs}
        self.lang_to_id = {lang: idx for idx, lang in enumerate(sorted({doc.lang for doc in self.all_docs.values()}))}
        self.similarity = self.build_similarity_matrix()
        self.pair_candidate_cache: dict[str, set[Tuple[str, str, str]]] = {}
        self.candidate_feature_columns: list[str] = []
        self.doc_feature_columns: list[str] = []
        self.oof_candidate_df: Optional[pd.DataFrame] = None
        self.oof_doc_df: Optional[pd.DataFrame] = None
        self.tuned_config: Optional[dict[str, Any]] = None
        self.final_models: dict[str, tuple[Optional[LGBMClassifier], Optional[float]]] = {}
        self.final_doc_gate: tuple[Optional[LGBMClassifier], Optional[float]] = (None, None)

    def build_alias_entries(self) -> List[tuple[str, str, str, regex.Pattern]]:
        concept_aliases: dict[str, set[str]] = defaultdict(set)
        concept_types: dict[str, str] = {}
        for doc in self.labeled_docs.values():
            for cluster in doc.clusters.values():
                concept_types[cluster.concept_key] = cluster.type
                for alias in cluster.aliases:
                    alias = clean_alias(alias)
                    if len(alias) >= 2:
                        concept_aliases[cluster.concept_key].add(alias)
        rows: list[tuple[int, str, str, str]] = []
        for concept_key, aliases in concept_aliases.items():
            entity_type = concept_types[concept_key]
            for alias in aliases:
                rows.append((len(alias), alias, concept_key, entity_type))
        rows = sorted(set(rows), key=lambda x: (-x[0], x[1], x[2]))
        return [(alias, concept_key, entity_type, alias_pattern(alias)) for _len_, alias, concept_key, entity_type in rows]

    def match_doc(self, doc_id: str) -> List[dict[str, Any]]:
        text = self.all_docs[doc_id].text
        text_cf = text.casefold()
        found: List[dict[str, Any]] = []
        for alias, concept_key, entity_type, pattern in self.alias_entries:
            if alias.casefold() not in text_cf:
                continue
            for match in pattern.finditer(text):
                found.append({
                    "start": match.start(),
                    "end": match.end(),
                    "text": text[match.start():match.end()],
                    "concept_key": concept_key,
                    "type": entity_type,
                })
        uniq = {(m["start"], m["end"], m["concept_key"]): m for m in found}
        mentions = list(uniq.values())
        mentions.sort(key=lambda m: (m["start"], m["end"] - m["start"], m["concept_key"]))
        return mentions

    def build_profile(self, doc_id: str) -> dict[str, Any]:
        doc = self.all_docs[doc_id]
        mentions: List[dict[str, Any]] = []
        for mention in self.matched_mentions[doc_id]:
            para_idx = None
            for idx, (start, end) in enumerate(doc.paragraphs):
                if start <= mention["start"] < mention["end"] <= end:
                    para_idx = idx
                    break
            sent_idx = None
            for idx, (start, end) in enumerate(doc.sent_spans):
                if start <= mention["start"] < mention["end"] <= end:
                    sent_idx = idx
                    break
            mentions.append({
                **mention,
                "para_idx": para_idx,
                "sent_idx": sent_idx,
                "in_title": any(start <= mention["start"] < mention["end"] <= end for start, end in doc.title_spans),
                "in_lead": any(start <= mention["start"] < mention["end"] <= end for start, end in doc.lead_spans),
            })
        by_concept: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for mention in mentions:
            by_concept[mention["concept_key"]].append(mention)
        concept_stats: dict[str, dict[str, Any]] = {}
        preferred_surface: dict[str, str] = {}
        for concept_key, items in by_concept.items():
            earliest = min(items, key=lambda m: m["start"])
            best = sorted(items, key=lambda m: (-int(m["in_title"]), -int(m["in_lead"]), -(m["end"] - m["start"]), m["start"]))[0]
            preferred_surface[concept_key] = best["text"]
            concept_stats[concept_key] = {
                "count": len(items),
                "first_pos": earliest["start"] / max(1, len(doc.text)),
                "title_count": sum(int(m["in_title"]) for m in items),
                "lead_count": sum(int(m["in_lead"]) for m in items),
                "max_len": max(m["end"] - m["start"] for m in items),
                "type": items[0]["type"],
            }
        return {"mentions": mentions, "by_concept": by_concept, "concept_stats": concept_stats, "preferred_surface": preferred_surface}

    def doc_repr_text(self, doc: DocInfo) -> str:
        pieces = [doc.title, doc.title, doc.title, doc.lead, doc.lead, doc.text, doc.url]
        return " ".join(piece for piece in pieces if piece)

    def doc_repr_concepts(self, doc_id: str) -> str:
        profile = self.profiles[doc_id]
        tokens: list[str] = []
        for concept_key, stats in profile["concept_stats"].items():
            token = concept_key.replace(":", "_").replace("|", " ")
            weight = 1 + 2 * int(stats["title_count"] > 0) + int(stats["lead_count"] > 0)
            tokens.extend([token] * weight)
        return " ".join(tokens)

    def build_similarity_matrix(self) -> pd.DataFrame:
        doc_ids = self.labeled_ids + self.test_ids
        text_corpus = [self.doc_repr_text(self.all_docs[doc_id]) for doc_id in doc_ids]
        concept_corpus = [self.doc_repr_concepts(doc_id) for doc_id in doc_ids]
        word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, lowercase=True, strip_accents="unicode", sublinear_tf=True)
        char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, lowercase=True, strip_accents="unicode", sublinear_tf=True)
        concept_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), min_df=1, lowercase=True, sublinear_tf=True)
        word_m = word_vec.fit_transform(text_corpus)
        char_m = char_vec.fit_transform(text_corpus)
        concept_m = concept_vec.fit_transform(concept_corpus)
        sim = 0.35 * cosine_similarity(word_m) + 0.35 * cosine_similarity(char_m) + 0.30 * cosine_similarity(concept_m)
        return pd.DataFrame(sim, index=doc_ids, columns=doc_ids)

    def build_priors(self, doc_ids: Iterable[str]) -> tuple[Counter, Counter, Counter, Counter, Counter]:
        pair_prior: Counter = Counter()
        subj_prior: Counter = Counter()
        obj_prior: Counter = Counter()
        concept_prior: Counter = Counter()
        pred_doc_prior: Counter = Counter()
        for doc_id in doc_ids:
            doc = self.labeled_docs[doc_id]
            seen_pred = set()
            seen_concepts = set()
            for predicate, sck, ock in doc.gold_edges:
                pair_prior[(predicate, sck, ock)] += 1
                subj_prior[(predicate, sck)] += 1
                obj_prior[(predicate, ock)] += 1
                seen_pred.add(predicate)
                seen_concepts.add(sck)
                seen_concepts.add(ock)
            for predicate in seen_pred:
                pred_doc_prior[predicate] += 1
            for concept_key in seen_concepts:
                concept_prior[concept_key] += 1
        return pair_prior, subj_prior, obj_prior, concept_prior, pred_doc_prior

    def retrieval_candidates(self, target_doc_id: str, source_doc_ids: Iterable[str], topk: int = 8) -> dict[Tuple[str, str, str], List[Tuple[float, int, str]]]:
        target_concepts = set(self.profiles[target_doc_id]["by_concept"])
        sims = []
        for source_doc_id in source_doc_ids:
            if source_doc_id == target_doc_id:
                continue
            sims.append((float(self.similarity.loc[target_doc_id, source_doc_id]), source_doc_id))
        sims.sort(key=lambda x: -x[0])
        candidates: dict[Tuple[str, str, str], List[Tuple[float, int, str]]] = {}
        rank = 0
        for sim_score, source_doc_id in sims[:topk]:
            rank += 1
            for edge in self.labeled_docs[source_doc_id].gold_edges:
                if edge[1] in target_concepts and edge[2] in target_concepts:
                    candidates.setdefault(edge, []).append((sim_score, rank, source_doc_id))
        for edge in candidates:
            candidates[edge].sort(key=lambda x: (-x[0], x[1]))
        return candidates

    def pair_candidates(self, doc_id: str) -> set[Tuple[str, str, str]]:
        if doc_id in self.pair_candidate_cache:
            return self.pair_candidate_cache[doc_id]
        mentions = self.profiles[doc_id]["mentions"]
        candidates: set[Tuple[str, str, str]] = set()
        for predicate, (subj_types, obj_types) in SCHEMA.items():
            subjects = [m for m in mentions if m["type"] in subj_types]
            objects = [m for m in mentions if m["type"] in obj_types]
            limit = 600 if predicate == "Dispersed_by" else 300
            for subject in subjects:
                for obj in objects:
                    if subject["concept_key"] == obj["concept_key"]:
                        continue
                    if subject["para_idx"] is None or subject["para_idx"] != obj["para_idx"]:
                        continue
                    dist = max(0, max(subject["start"], obj["start"]) - min(subject["end"], obj["end"]))
                    if dist <= limit:
                        candidates.add((predicate, subject["concept_key"], obj["concept_key"]))
        self.pair_candidate_cache[doc_id] = candidates
        return candidates

    def pair_proximity(self, doc_id: str, sck: str, ock: str) -> dict[str, Any]:
        profile = self.profiles[doc_id]
        subjects = profile["by_concept"].get(sck, [])
        objects = profile["by_concept"].get(ock, [])
        if not subjects or not objects:
            return {
                "same_sent": 0,
                "same_para": 0,
                "same_title": 0,
                "same_lead": 0,
                "any_title": 0,
                "any_lead": 0,
                "min_char_dist": 9999,
                "min_para_gap": 99,
            }
        same_sent = same_para = same_title = same_lead = 0
        min_char = 9999
        min_gap = 99
        for subject in subjects:
            for obj in objects:
                if subject["start"] == obj["start"] and subject["end"] == obj["end"]:
                    continue
                dist = max(0, max(subject["start"], obj["start"]) - min(subject["end"], obj["end"]))
                min_char = min(min_char, dist)
                if subject["sent_idx"] is not None and subject["sent_idx"] == obj["sent_idx"]:
                    same_sent = 1
                if subject["para_idx"] is not None and subject["para_idx"] == obj["para_idx"]:
                    same_para = 1
                if subject["para_idx"] is not None and obj["para_idx"] is not None:
                    min_gap = min(min_gap, abs(subject["para_idx"] - obj["para_idx"]))
                if subject["in_title"] and obj["in_title"]:
                    same_title = 1
                if subject["in_lead"] and obj["in_lead"]:
                    same_lead = 1
        return {
            "same_sent": same_sent,
            "same_para": same_para,
            "same_title": same_title,
            "same_lead": same_lead,
            "any_title": int(any(m["in_title"] for m in subjects) or any(m["in_title"] for m in objects)),
            "any_lead": int(any(m["in_lead"] for m in subjects) or any(m["in_lead"] for m in objects)),
            "min_char_dist": min_char if min_char < 9999 else 9999,
            "min_para_gap": min_gap if min_gap < 99 else 99,
        }

    def build_candidate_rows(self, target_doc_ids: Iterable[str], source_doc_ids: Iterable[str], priors: tuple[Counter, Counter, Counter, Counter, Counter], include_labels: bool) -> pd.DataFrame:
        pair_prior, subj_prior, obj_prior, concept_prior, pred_doc_prior = priors
        rows: List[dict[str, Any]] = []
        for doc_id in target_doc_ids:
            doc = self.all_docs[doc_id]
            profile = self.profiles[doc_id]
            retrieval = self.retrieval_candidates(doc_id, source_doc_ids, topk=8)
            pair_cands = self.pair_candidates(doc_id)
            edges = set(retrieval) | pair_cands
            for predicate, sck, ock in edges:
                sstats = profile["concept_stats"].get(sck, {"count": 0, "first_pos": 1.0, "title_count": 0, "lead_count": 0, "max_len": 0})
                ostats = profile["concept_stats"].get(ock, {"count": 0, "first_pos": 1.0, "title_count": 0, "lead_count": 0, "max_len": 0})
                prox = self.pair_proximity(doc_id, sck, ock)
                supports = retrieval.get((predicate, sck, ock), [])
                sims = [entry[0] for entry in supports]
                ranks = [entry[1] for entry in supports]
                row = {
                    "doc_id": doc_id,
                    "predicate": predicate,
                    "sck": sck,
                    "ock": ock,
                    "label": int((predicate, sck, ock) in getattr(doc, "gold_edges", set())) if include_labels else -1,
                    "from_pair": int((predicate, sck, ock) in pair_cands),
                    "from_ret": int((predicate, sck, ock) in retrieval),
                    "ret_count": len(supports),
                    "ret_sum": float(sum(sims)) if sims else 0.0,
                    "ret_max": float(max(sims)) if sims else 0.0,
                    "ret_mean": float(sum(sims) / len(sims)) if sims else 0.0,
                    "ret_top1": float(sims[0]) if sims else 0.0,
                    "ret_top2": float(sims[1]) if len(sims) > 1 else 0.0,
                    "ret_rank_min": min(ranks) if ranks else 99,
                    "ret_lang_match_count": sum(int(self.all_docs[source].lang == doc.lang) for _sim, _rank, source in supports),
                    "pair_pred_prior": pair_prior[(predicate, sck, ock)],
                    "subj_pred_prior": subj_prior[(predicate, sck)],
                    "obj_pred_prior": obj_prior[(predicate, ock)],
                    "subj_doc_prior": concept_prior[sck],
                    "obj_doc_prior": concept_prior[ock],
                    "pred_doc_prior": pred_doc_prior[predicate],
                    "subj_count": sstats["count"],
                    "obj_count": ostats["count"],
                    "subj_first_pos": sstats["first_pos"],
                    "obj_first_pos": ostats["first_pos"],
                    "subj_title": sstats["title_count"],
                    "obj_title": ostats["title_count"],
                    "subj_lead": sstats["lead_count"],
                    "obj_lead": ostats["lead_count"],
                    "subj_maxlen": sstats["max_len"],
                    "obj_maxlen": ostats["max_len"],
                    "lang_id": self.lang_to_id.get(doc.lang, 0),
                }
                row.update(prox)
                rows.append(row)
        frame = pd.DataFrame(rows)
        if frame.empty:
            frame = pd.DataFrame(columns=["doc_id", "predicate", "sck", "ock", "label"])
        return frame

    def build_doc_rows(self, target_doc_ids: Iterable[str], source_doc_ids: Iterable[str]) -> pd.DataFrame:
        source_doc_ids = list(source_doc_ids)
        empty_source_ids = [doc_id for doc_id in source_doc_ids if not self.labeled_docs[doc_id].gold_edges]
        nonempty_source_ids = [doc_id for doc_id in source_doc_ids if self.labeled_docs[doc_id].gold_edges]
        rows = []
        for doc_id in target_doc_ids:
            doc = self.all_docs[doc_id]
            profile = self.profiles[doc_id]
            mention_counts = Counter(m["type"] for m in profile["mentions"])
            retrieval = self.retrieval_candidates(doc_id, source_doc_ids, topk=8)
            pair_cands = self.pair_candidates(doc_id)
            sims = sorted([float(self.similarity.loc[doc_id, src]) for src in source_doc_ids if src != doc_id], reverse=True)
            empty_sims = sorted([float(self.similarity.loc[doc_id, src]) for src in empty_source_ids if src != doc_id], reverse=True)
            nonempty_sims = sorted([float(self.similarity.loc[doc_id, src]) for src in nonempty_source_ids if src != doc_id], reverse=True)
            rows.append({
                "doc_id": doc_id,
                "label": int(bool(getattr(doc, "gold_edges", set()))) if doc_id in self.labeled_docs else -1,
                "lang_id": self.lang_to_id.get(doc.lang, 0),
                "title_len": len(doc.title),
                "lead_len": len(doc.lead),
                "text_len": len(doc.text),
                "paragraph_count": len(doc.paragraphs),
                "matched_total": len(profile["mentions"]),
                "matched_concepts": len(profile["by_concept"]),
                "matched_pest": mention_counts["Pest"],
                "matched_plant": mention_counts["Plant"],
                "matched_disease": mention_counts["Disease"],
                "matched_vector": mention_counts["Vector"],
                "matched_pathway": mention_counts["Dissemination_pathway"],
                "matched_location": mention_counts["Location"],
                "matched_date": mention_counts["Date"],
                "title_concepts": sum(int(v["title_count"] > 0) for v in profile["concept_stats"].values()),
                "lead_concepts": sum(int(v["lead_count"] > 0) for v in profile["concept_stats"].values()),
                "ret_candidate_edges": len(retrieval),
                "pair_candidate_edges": len(pair_cands),
                "top_sim_1": sims[0] if sims else 0.0,
                "top_sim_3_mean": float(np.mean(sims[:3])) if sims else 0.0,
                "top_nonempty_sim": nonempty_sims[0] if nonempty_sims else 0.0,
                "top_empty_sim": empty_sims[0] if empty_sims else 0.0,
            })
        frame = pd.DataFrame(rows)
        return frame

    def train_binary_model(self, train_df: pd.DataFrame, feature_cols: List[str], label_col: str, *, is_doc_gate: bool = False) -> tuple[Optional[LGBMClassifier], Optional[float]]:
        if train_df.empty:
            return None, 0.0
        y = train_df[label_col].astype(int).values
        if y.sum() == 0:
            return None, 0.0
        if y.sum() == len(y):
            return None, 1.0
        model = LGBMClassifier(
            objective="binary",
            n_estimators=160 if not is_doc_gate else 120,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=10 if not is_doc_gate else 5,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
            n_jobs=1,
            force_col_wise=True,
        )
        model.fit(train_df[feature_cols], y)
        return model, None

    def predict_binary_model(self, model: Optional[LGBMClassifier], const: Optional[float], frame: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        if frame.empty:
            return np.array([], dtype=float)
        if const is not None:
            return np.full(len(frame), float(const), dtype=float)
        assert model is not None
        return model.predict_proba(frame[feature_cols])[:, 1]

    @staticmethod
    def doc_f1(pred: set[Tuple[str, str, str]], gold: set[Tuple[str, str, str]]) -> float:
        if not pred and not gold:
            return 1.0
        if not pred or not gold:
            return 0.0
        tp = len(pred & gold)
        precision = tp / len(pred) if pred else 0.0
        recall = tp / len(gold) if gold else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def run_cv(self, folds: int = 5, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
        oof_candidate_parts = []
        oof_doc_parts = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.labeled_ids), start=1):
            train_ids = [self.labeled_ids[i] for i in train_idx]
            val_ids = [self.labeled_ids[i] for i in val_idx]
            priors = self.build_priors(train_ids)
            train_cand = self.build_candidate_rows(train_ids, train_ids, priors, include_labels=True)
            val_cand = self.build_candidate_rows(val_ids, train_ids, priors, include_labels=True)
            train_doc = self.build_doc_rows(train_ids, train_ids)
            val_doc = self.build_doc_rows(val_ids, train_ids)

            if not self.candidate_feature_columns:
                self.candidate_feature_columns = [c for c in train_cand.columns if c not in {"doc_id", "predicate", "sck", "ock", "label"}]
            if not self.doc_feature_columns:
                self.doc_feature_columns = [c for c in train_doc.columns if c not in {"doc_id", "label"}]

            doc_model, doc_const = self.train_binary_model(train_doc, self.doc_feature_columns, "label", is_doc_gate=True)
            val_doc = val_doc.copy()
            val_doc["doc_prob"] = self.predict_binary_model(doc_model, doc_const, val_doc, self.doc_feature_columns)
            oof_doc_parts.append(val_doc)

            val_cand = val_cand.copy()
            val_cand["prob"] = 0.0
            for predicate in PREDICATES:
                tr = train_cand[train_cand["predicate"] == predicate].copy()
                va = val_cand[val_cand["predicate"] == predicate].copy()
                if va.empty:
                    continue
                model, const = self.train_binary_model(tr, self.candidate_feature_columns, "label", is_doc_gate=False)
                val_cand.loc[val_cand["predicate"] == predicate, "prob"] = self.predict_binary_model(model, const, va, self.candidate_feature_columns)
            oof_candidate_parts.append(val_cand)
            print(f"[cv] fold={fold} train_docs={len(train_ids)} val_docs={len(val_ids)} train_edges={len(train_cand)} val_edges={len(val_cand)}")

        self.oof_candidate_df = pd.concat(oof_candidate_parts, ignore_index=True)
        self.oof_doc_df = pd.concat(oof_doc_parts, ignore_index=True)
        return self.oof_candidate_df, self.oof_doc_df

    def apply_config(self, candidate_df: pd.DataFrame, doc_df: pd.DataFrame, gold_lookup: Optional[dict[str, set[Tuple[str, str, str]]]], config: dict[str, Any]) -> tuple[dict[str, set[Tuple[str, str, str]]], float]:
        doc_keep = {row.doc_id: float(row.doc_prob) >= config["doc_gate_threshold"] for row in doc_df.itertuples()}
        preds: dict[str, set[Tuple[str, str, str]]] = defaultdict(set)
        for predicate in PREDICATES:
            sub = candidate_df[candidate_df["predicate"] == predicate].copy()
            if sub.empty:
                continue
            sub = sub[sub["prob"] >= config["predicate_thresholds"][predicate]].copy()
            if sub.empty:
                continue
            sub.sort_values(["doc_id", "prob"], ascending=[True, False], inplace=True)
            cap = config["predicate_caps"][predicate]
            if cap < 999999:
                sub = sub.groupby("doc_id", group_keys=False).head(cap)
            for row in sub.itertuples():
                if doc_keep.get(row.doc_id, True):
                    preds[row.doc_id].add((row.predicate, row.sck, row.ock))
        score = float("nan")
        if gold_lookup is not None:
            score = float(np.mean([self.doc_f1(preds.get(doc_id, set()), gold_lookup[doc_id]) for doc_id in gold_lookup]))
        return preds, score

    def tune_config(self) -> dict[str, Any]:
        assert self.oof_candidate_df is not None and self.oof_doc_df is not None
        gold_lookup = {doc_id: self.labeled_docs[doc_id].gold_edges for doc_id in self.labeled_ids}
        config = {
            "doc_gate_threshold": 0.45,
            "predicate_thresholds": {predicate: 0.5 for predicate in PREDICATES},
            "predicate_caps": {
                "Located_in": 8,
                "Found_on": 4,
                "Occurs_on": 4,
                "Causes": 3,
                "Affects": 3,
                "Dispersed_by": 4,
                "Transmits": 2,
            },
        }
        threshold_grids = {
            "doc_gate_threshold": [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75],
            "Located_in": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50],
            "Found_on": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50],
            "Occurs_on": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50],
            "Causes": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
            "Affects": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
            "Dispersed_by": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
            "Transmits": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
        }
        cap_grids = {
            "Located_in": [5, 8, 12, 20, 999999],
            "Found_on": [2, 4, 6, 999999],
            "Occurs_on": [2, 4, 6, 999999],
            "Causes": [1, 2, 3, 999999],
            "Affects": [1, 2, 3, 999999],
            "Dispersed_by": [1, 2, 4, 999999],
            "Transmits": [1, 2, 999999],
        }
        _preds, best_score = self.apply_config(self.oof_candidate_df, self.oof_doc_df, gold_lookup, config)
        for _round in range(3):
            improved = False
            for gate_threshold in threshold_grids["doc_gate_threshold"]:
                trial = json.loads(json.dumps(config))
                trial["doc_gate_threshold"] = gate_threshold
                _preds, score = self.apply_config(self.oof_candidate_df, self.oof_doc_df, gold_lookup, trial)
                if score > best_score:
                    config, best_score = trial, score
                    improved = True
            for predicate in PREDICATES:
                for threshold in threshold_grids[predicate]:
                    trial = json.loads(json.dumps(config))
                    trial["predicate_thresholds"][predicate] = threshold
                    _preds, score = self.apply_config(self.oof_candidate_df, self.oof_doc_df, gold_lookup, trial)
                    if score > best_score:
                        config, best_score = trial, score
                        improved = True
                for cap in cap_grids[predicate]:
                    trial = json.loads(json.dumps(config))
                    trial["predicate_caps"][predicate] = cap
                    _preds, score = self.apply_config(self.oof_candidate_df, self.oof_doc_df, gold_lookup, trial)
                    if score > best_score:
                        config, best_score = trial, score
                        improved = True
            if not improved:
                break
        config["cv_macro_f1"] = best_score
        self.tuned_config = config
        return config

    def fit_final_models(self) -> None:
        priors = self.build_priors(self.labeled_ids)
        train_cand = self.build_candidate_rows(self.labeled_ids, self.labeled_ids, priors, include_labels=True)
        train_doc = self.build_doc_rows(self.labeled_ids, self.labeled_ids)
        if not self.candidate_feature_columns:
            self.candidate_feature_columns = [c for c in train_cand.columns if c not in {"doc_id", "predicate", "sck", "ock", "label"}]
        if not self.doc_feature_columns:
            self.doc_feature_columns = [c for c in train_doc.columns if c not in {"doc_id", "label"}]
        self.final_doc_gate = self.train_binary_model(train_doc, self.doc_feature_columns, "label", is_doc_gate=True)
        self.final_models = {}
        for predicate in PREDICATES:
            tr = train_cand[train_cand["predicate"] == predicate].copy()
            self.final_models[predicate] = self.train_binary_model(tr, self.candidate_feature_columns, "label", is_doc_gate=False)

    def predict_candidates(self, target_doc_ids: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        priors = self.build_priors(self.labeled_ids)
        cand_df = self.build_candidate_rows(target_doc_ids, self.labeled_ids, priors, include_labels=False)
        doc_df = self.build_doc_rows(target_doc_ids, self.labeled_ids)
        cand_df = cand_df.copy()
        doc_df = doc_df.copy()
        doc_model, doc_const = self.final_doc_gate
        doc_df["doc_prob"] = self.predict_binary_model(doc_model, doc_const, doc_df, self.doc_feature_columns)
        cand_df["prob"] = 0.0
        for predicate in PREDICATES:
            sub = cand_df[cand_df["predicate"] == predicate].copy()
            if sub.empty:
                continue
            model, const = self.final_models[predicate]
            cand_df.loc[cand_df["predicate"] == predicate, "prob"] = self.predict_binary_model(model, const, sub, self.candidate_feature_columns)
        return cand_df, doc_df

    def concept_edges_to_string_edges(self, doc_id: str, edges: set[Tuple[str, str, str]]) -> List[dict[str, str]]:
        profile = self.profiles[doc_id]
        string_edges = []
        seen = set()
        for predicate, sck, ock in sorted(edges):
            subject = profile["preferred_surface"].get(sck)
            object_ = profile["preferred_surface"].get(ock)
            if not subject or not object_:
                continue
            key = (predicate, subject, object_)
            if key in seen:
                continue
            seen.add(key)
            string_edges.append({"predicate": predicate, "subject": subject, "object": object_})
        return string_edges

    def make_diverse_config(self, base: dict[str, Any]) -> dict[str, Any]:
        diverse = json.loads(json.dumps(base))
        diverse["doc_gate_threshold"] = max(0.05, base["doc_gate_threshold"] - 0.10)
        for predicate in ["Located_in", "Found_on", "Occurs_on", "Causes", "Affects"]:
            diverse["predicate_thresholds"][predicate] = max(0.05, base["predicate_thresholds"][predicate] - 0.05)
        for predicate in PREDICATES:
            if diverse["predicate_caps"][predicate] < 999999:
                diverse["predicate_caps"][predicate] = int(diverse["predicate_caps"][predicate] * 1.5) if diverse["predicate_caps"][predicate] >= 2 else 2
        return diverse

    def make_safeplus_config(self, base: dict[str, Any]) -> dict[str, Any]:
        safe = json.loads(json.dumps(base))
        safe["doc_gate_threshold"] = min(0.95, base["doc_gate_threshold"] + 0.05)
        for predicate in PREDICATES:
            safe["predicate_thresholds"][predicate] = min(0.95, base["predicate_thresholds"][predicate] + 0.05)
        for predicate in PREDICATES:
            if safe["predicate_caps"][predicate] < 999999:
                safe["predicate_caps"][predicate] = max(1, int(math.ceil(safe["predicate_caps"][predicate] * 0.75)))
        return safe

    def edges_from_predictions(self, cand_df: pd.DataFrame, doc_df: pd.DataFrame, config: dict[str, Any]) -> dict[str, List[dict[str, str]]]:
        concept_preds, _score = self.apply_config(cand_df, doc_df, gold_lookup=None, config=config)
        return {doc_id: self.concept_edges_to_string_edges(doc_id, concept_preds.get(doc_id, set())) for doc_id in [*self.test_ids]}

    def read_reference_submission(self, path: str | Path) -> dict[str, List[dict[str, str]]]:
        out = {}
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out[str(row["doc_id"])] = json.loads(row["knowledge_graph"])
        return out

    def merge_additions(self, base_pred: dict[str, List[dict[str, str]]], add_pred: dict[str, List[dict[str, str]]]) -> dict[str, List[dict[str, str]]]:
        merged = {}
        for doc_id in self.test_ids:
            edges = list(base_pred.get(doc_id, []))
            seen = {(e["predicate"], e["subject"], e["object"]) for e in edges}
            for edge in add_pred.get(doc_id, []):
                key = (edge["predicate"], edge["subject"], edge["object"])
                if key not in seen:
                    seen.add(key)
                    edges.append(edge)
            merged[doc_id] = edges
        return merged

    def keep_high_confidence_additions(self, cand_df: pd.DataFrame, threshold: float = 0.75, support_min: int = 1) -> dict[str, List[dict[str, str]]]:
        out: dict[str, List[dict[str, str]]] = defaultdict(list)
        for row in cand_df.itertuples():
            if float(row.prob) < threshold or int(row.ret_count) < support_min:
                continue
            subject = self.profiles[row.doc_id]["preferred_surface"].get(row.sck)
            object_ = self.profiles[row.doc_id]["preferred_surface"].get(row.ock)
            if not subject or not object_:
                continue
            out[row.doc_id].append({"predicate": row.predicate, "subject": subject, "object": object_})
        deduped = {}
        for doc_id, edges in out.items():
            seen = set()
            clean = []
            for edge in sorted(edges, key=lambda e: (e["predicate"], e["subject"], e["object"])):
                key = (edge["predicate"], edge["subject"], edge["object"])
                if key not in seen:
                    seen.add(key)
                    clean.append(edge)
            deduped[doc_id] = clean
        return deduped

    def write_submission(self, predictions: dict[str, List[dict[str, str]]], output_csv: str | Path) -> None:
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["doc_id", "knowledge_graph"])
            for doc_id in self.test_ids:
                edges = predictions.get(doc_id, [])
                writer.writerow([doc_id, json.dumps(edges, ensure_ascii=False)])

    def compare_to_reference(self, reference: dict[str, List[dict[str, str]]], preds: dict[str, List[dict[str, str]]]) -> pd.DataFrame:
        rows = []
        for doc_id in self.test_ids:
            ref = {(e["predicate"], e["subject"], e["object"]) for e in reference.get(doc_id, [])}
            new = {(e["predicate"], e["subject"], e["object"]) for e in preds.get(doc_id, [])}
            rows.append({
                "doc_id": doc_id,
                "ref_edges": len(ref),
                "new_edges": len(new),
                "added": len(new - ref),
                "removed": len(ref - new),
            })
        return pd.DataFrame(rows).sort_values(["added", "removed", "doc_id"], ascending=[False, False, True])

    def build_report(self, output_dir: str | Path, safe_pred: dict[str, List[dict[str, str]]], diverse_pred: dict[str, List[dict[str, str]]], safeplus_pred: dict[str, List[dict[str, str]]], addon_pred: Optional[dict[str, List[dict[str, str]]]] = None) -> dict[str, Any]:
        output_dir = Path(output_dir)
        report = {
            "num_labeled_docs": len(self.labeled_ids),
            "num_test_docs": len(self.test_ids),
            "candidate_cv_rows": int(len(self.oof_candidate_df)) if self.oof_candidate_df is not None else None,
            "doc_cv_rows": int(len(self.oof_doc_df)) if self.oof_doc_df is not None else None,
            "tuned_config": self.tuned_config,
            "safe_edge_total": int(sum(len(v) for v in safe_pred.values())),
            "diverse_edge_total": int(sum(len(v) for v in diverse_pred.values())),
            "safeplus_edge_total": int(sum(len(v) for v in safeplus_pred.values())),
            "addon_edge_total": int(sum(len(v) for v in addon_pred.values())) if addon_pred is not None else None,
        }
        with open(output_dir / "report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description="PestCLEF 2026 major revamp hybrid extractor")
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--dev-json", required=True)
    parser.add_argument("--test-json", required=True)
    parser.add_argument("--texts-zip", required=True)
    parser.add_argument("--reference-csv", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train_docs = load_json(args.train_json)
    dev_docs = load_json(args.dev_json)
    test_docs = load_json(args.test_json)

    model = MajorRevamp(train_docs, dev_docs, test_docs, args.texts_zip, reference_csv=args.reference_csv)

    # gold reconstruction sanity check
    mismatch_docs = []
    for raw_doc in train_docs + dev_docs:
        entities, clusters, find = build_union_find(raw_doc)
        rebuilt = set()
        cluster_map = {}
        for root, ids in clusters.items():
            member_entities = [entities[eid] for eid in ids]
            entity_type = member_entities[0]["type"]
            aliases = tuple(sorted({clean_alias(entity["form"]) for entity in member_entities}))
            norms = tuple(sorted({norm for entity in member_entities for norm in norm_tuple(entity)}))
            cluster_map[root] = make_concept_key(entity_type, norms, aliases)
        for rel in raw_doc.get("text_bound_annotations", {}).get("relations", []):
            sarg, oarg = ARG_MAP[rel["type"]]
            rebuilt.add((rel["type"], cluster_map[find(rel["args"][sarg])], cluster_map[find(rel["args"][oarg])]))
        original = set()
        doc_id = str(raw_doc["doc_id"])
        if doc_id in model.labeled_docs:
            original = model.labeled_docs[doc_id].gold_edges
        if rebuilt != original:
            mismatch_docs.append(doc_id)
    print(f"[audit] reconstruction_mismatches={len(mismatch_docs)}")

    oof_candidate_df, oof_doc_df = model.run_cv(folds=args.cv_folds, seed=42)
    tuned = model.tune_config()
    print(f"[cv] tuned_macro_f1={tuned['cv_macro_f1']:.6f}")
    print(f"[cv] doc_gate_threshold={tuned['doc_gate_threshold']}")
    print(f"[cv] predicate_thresholds={tuned['predicate_thresholds']}")
    print(f"[cv] predicate_caps={tuned['predicate_caps']}")

    model.fit_final_models()
    test_cand_df, test_doc_df = model.predict_candidates(model.test_ids)
    safe_pred = model.edges_from_predictions(test_cand_df, test_doc_df, tuned)
    diverse_config = model.make_diverse_config(tuned)
    safeplus_config = model.make_safeplus_config(tuned)
    diverse_pred = model.edges_from_predictions(test_cand_df, test_doc_df, diverse_config)
    safeplus_pred = model.edges_from_predictions(test_cand_df, test_doc_df, safeplus_config)

    out_dir = Path(args.output_dir)
    safe_csv = out_dir / "submission_v30_major_revamp_safe.csv"
    diverse_csv = out_dir / "submission_v30_major_revamp_diverse.csv"
    safeplus_csv = out_dir / "submission_v30_major_revamp_safeplus.csv"
    model.write_submission(safe_pred, safe_csv)
    model.write_submission(diverse_pred, diverse_csv)
    model.write_submission(safeplus_pred, safeplus_csv)

    addon_pred = None
    if args.reference_csv:
        reference = model.read_reference_submission(args.reference_csv)
        high_conf_additions = model.keep_high_confidence_additions(test_cand_df, threshold=max(0.75, tuned['predicate_thresholds']['Located_in'] + 0.25), support_min=1)
        addon_pred = model.merge_additions(reference, high_conf_additions)
        addon_csv = out_dir / "submission_v30_major_revamp_addon_on_reference.csv"
        model.write_submission(addon_pred, addon_csv)
        delta_df = model.compare_to_reference(reference, safe_pred)
        delta_df.to_csv(out_dir / "safe_vs_reference_delta.csv", index=False)
        delta_df2 = model.compare_to_reference(reference, addon_pred)
        delta_df2.to_csv(out_dir / "addon_vs_reference_delta.csv", index=False)

    model.build_report(out_dir, safe_pred, diverse_pred, safeplus_pred, addon_pred)

    # candidate docs diagnostic
    top_changes = test_cand_df.sort_values("prob", ascending=False).head(500)
    top_changes.to_csv(out_dir / "top_test_candidate_scores.csv", index=False)

    # zip outputs
    zip_path = out_dir / "pestclef_major_revamp_outputs.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.glob("*")):
            if path.name == zip_path.name:
                continue
            zf.write(path, arcname=path.name)

    print(f"[done] outputs={out_dir}")
    print(f"[done] safe={safe_csv}")
    print(f"[done] diverse={diverse_csv}")
    print(f"[done] safeplus={safeplus_csv}")
    if args.reference_csv:
        print(f"[done] addon={out_dir / 'submission_v30_major_revamp_addon_on_reference.csv'}")
        print(f"[done] reference={args.reference_csv}")
    print(f"[done] zip={zip_path}")


if __name__ == "__main__":
    main()
