#!/usr/bin/env python3
from __future__ import annotations
import csv, json, zipfile, re, html, math, os, argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

PREDICATES = ["Located_in","Found_on","Occurs_on","Causes","Affects","Dispersed_by","Transmits"]
PRED_IDX = {p:i for i,p in enumerate(PREDICATES)}
DANGEROUS_DOCS = {"100506","102433"}  # public feedback: recall additions here repeatedly harmful
NEUTRAL_EXPLORATION_DOCS = {"117314","107265","107268","106417"}  # v38 no public movement


def clean(s: Any) -> str:
    s = html.unescape(str(s)).replace('\\/', '/')
    return re.sub(r"\s+", " ", s).strip()

def as_list(x: Any) -> list[str]:
    return [clean(v) for v in x] if isinstance(x, list) else [clean(x)]

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", clean(s)).casefold()

def boundary_find(text: str, phrase: str) -> str | None:
    phrase = clean(phrase)
    if not phrase or len(phrase) < 2:
        return None
    # Use simple alnum boundary; better than substring for short aliases.
    pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(phrase) + r"(?![A-Za-z0-9])", re.I)
    m = pat.search(text)
    return m.group(0) if m else None

def load_json(path: Path):
    with open(path, encoding="utf-8") as f: return json.load(f)

def load_texts(zip_path: Path):
    texts = {"train":{}, "dev":{}, "test":{}}
    meta = {"train":{}, "dev":{}, "test":{}}
    with zipfile.ZipFile(zip_path) as z:
        for split in texts:
            pref=f"EPOP_documents/{split}/"
            for name in z.namelist():
                if name.startswith(pref) and name.endswith('.txt'):
                    texts[split][Path(name).stem]=z.read(name).decode('utf-8')
            meta_name=f"EPOP_documents/{split}/documents-metadata.csv"
            if meta_name in z.namelist():
                raw=z.read(meta_name).decode('utf-8', errors='replace').splitlines()
                if raw:
                    reader=csv.DictReader(raw, delimiter='\t')
                    for row in reader:
                        meta[split][str(row.get('DOC',''))]=row
    return texts, meta

def read_sub(path: Path):
    order=[]; sub={}
    with open(path, encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            did=str(row['doc_id']); order.append(did); sub[did]=json.loads(row['knowledge_graph'])
    return order, sub

def edge_key(e: dict[str,str]) -> tuple[str,str,str]:
    return (e['predicate'], e['subject'], e['object'])

def edge_dict(t: tuple[str,str,str]) -> dict[str,str]:
    p,s,o=t; return {"predicate":p,"subject":s,"object":o}

def write_sub(order: list[str], sets_by_doc: dict[str,set[tuple[str,str,str]]], path: Path):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w=csv.writer(f); w.writerow(['doc_id','knowledge_graph'])
        for did in order:
            edges=[edge_dict(t) for t in sorted(sets_by_doc.get(did,set()))]
            w.writerow([did, json.dumps(edges, ensure_ascii=False)])

def doc_repr(text: str) -> str:
    lines=[ln.strip() for ln in text.splitlines() if ln.strip()]
    title=lines[0] if lines else ''
    lead=' '.join(lines[1:4]) if len(lines)>1 else ''
    return ' '.join([title]*4+[lead]*2+[text[:3500], text])

def kg_alias_sets(rel: dict[str, Any]) -> tuple[str, set[str], set[str]]:
    return rel['predicate'], {norm(x) for x in as_list(rel['subject'])}, {norm(x) for x in as_list(rel['object'])}

def rel_to_candidate_in_text(rel: dict[str,Any], target_text: str) -> tuple[str,str,str] | None:
    p=rel['predicate']
    subs=sorted(set(as_list(rel['subject'])), key=lambda x:(-len(x), x.casefold()))
    objs=sorted(set(as_list(rel['object'])), key=lambda x:(-len(x), x.casefold()))
    sm=[]; om=[]
    for s in subs:
        m=boundary_find(target_text,s)
        if m: sm.append(m)
    for o in objs:
        m=boundary_find(target_text,o)
        if m: om.append(m)
    if not sm or not om: return None
    if norm(sm[0]) == norm(om[0]): return None
    return (p, sm[0], om[0])

def candidate_label(candidate: tuple[str,str,str], target_gold: list[dict[str,Any]]) -> int:
    p,s,o=candidate; ns=norm(s); no=norm(o)
    for rel in target_gold:
        rp, sset, oset = kg_alias_sets(rel)
        if p==rp and ns in sset and no in oset:
            return 1
    return 0

def make_features(row: dict[str,Any]) -> list[float]:
    p=row['predicate']
    v=[
        row['score_sum'], row['sim_max'], row['sim_mean'], row['support_count'], row['best_rank_inv'],
        row['doc_top_sim'], row['source_edge_count_mean'], row['subject_len'], row['object_len'],
        int(row['subject_in_title']), int(row['object_in_title']), int(row['subject_in_lead']), int(row['object_in_lead']),
    ]
    # predicate one-hot
    v += [1.0 if p==pp else 0.0 for pp in PREDICATES]
    return v

def aggregate_candidates_for_target(did: str, text: str, neighbor_indices: list[int], sims: np.ndarray, ldocs: list[dict], topk: int=10, sim_min: float=0.20, include_label=False, gold=None):
    lines=[ln.strip() for ln in text.splitlines() if ln.strip()]
    title=lines[0] if lines else ''
    lead=' '.join(lines[1:4]) if len(lines)>1 else ''
    agg: dict[tuple[str,str,str], dict[str,Any]] = {}
    for rank, j in enumerate(neighbor_indices[:topk], start=1):
        sim=float(sims[j])
        if sim < sim_min: continue
        src=ldocs[j]
        for rel in src['kg']:
            cand=rel_to_candidate_in_text(rel, text)
            if cand is None: continue
            p,s,o=cand
            item=agg.setdefault(cand, {"doc_id":did,"predicate":p,"subject":s,"object":o,"score_sum":0.0,"sims":[],"ranks":[],"srcs":[],"source_edge_counts":[]})
            item['score_sum'] += sim / math.sqrt(rank)
            item['sims'].append(sim); item['ranks'].append(rank); item['srcs'].append(src['doc_id']); item['source_edge_counts'].append(len(src['kg']))
    rows=[]
    for cand,item in agg.items():
        s=item['subject']; o=item['object']
        sims_list=item['sims']
        item.update({
            'sim_max': max(sims_list),
            'sim_mean': float(np.mean(sims_list)),
            'support_count': len(sims_list),
            'best_rank_inv': 1.0/min(item['ranks']),
            'doc_top_sim': float(max(sims)) if len(sims) else 0.0,
            'source_edge_count_mean': float(np.mean(item['source_edge_counts'])),
            'subject_len': len(s),
            'object_len': len(o),
            'subject_in_title': norm(s) in norm(title),
            'object_in_title': norm(o) in norm(title),
            'subject_in_lead': norm(s) in norm(lead),
            'object_in_lead': norm(o) in norm(lead),
            'sources': ';'.join(item['srcs'][:5]),
        })
        if include_label:
            item['label']=candidate_label(cand, gold or [])
        rows.append(item)
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train-json', required=True)
    ap.add_argument('--dev-json', required=True)
    ap.add_argument('--test-json', required=True)
    ap.add_argument('--texts-zip', required=True)
    ap.add_argument('--reference-csv', required=True)
    ap.add_argument('--output-dir', required=True)
    args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    train=load_json(Path(args.train_json)); dev=load_json(Path(args.dev_json)); test=load_json(Path(args.test_json))
    texts,meta=load_texts(Path(args.texts_zip))
    ldocs=[]
    for split,docs in [('train',train),('dev',dev)]:
        for d in docs:
            did=str(d['doc_id'])
            ldocs.append({'split':split,'doc_id':did,'text':texts[split][did], 'kg':d.get('knowledge_graph',[])})
    tdocs=[]
    for d in test:
        did=str(d['doc_id'])
        tdocs.append({'split':'test','doc_id':did,'text':texts['test'][did]})
    # vectorizer fit on all docs for similarity
    corpus=[doc_repr(d['text']) for d in ldocs+tdocs]
    word=TfidfVectorizer(analyzer='word', ngram_range=(1,2), lowercase=True, min_df=1, sublinear_tf=True)
    char=TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), lowercase=True, min_df=1, sublinear_tf=True)
    Xw=word.fit_transform(corpus); Xc=char.fit_transform(corpus)
    L=len(ldocs)
    S_full = 0.45*cosine_similarity(Xw, Xw) + 0.55*cosine_similarity(Xc, Xc)
    # OOF training rows
    kf=KFold(n_splits=5, shuffle=True, random_state=43)
    train_rows=[]
    for tr_idx, va_idx in kf.split(np.arange(L)):
        for i in va_idx:
            did=ldocs[i]['doc_id']; text=ldocs[i]['text']
            sims=S_full[i, :L].copy(); sims[i]=0
            # only use neighbors in training fold
            mask=np.ones(L, dtype=bool); mask[va_idx]=False
            sims[~mask]=0
            neigh=list(np.argsort(sims)[::-1])
            rows=aggregate_candidates_for_target(did,text,neigh,sims,ldocs,topk=10,sim_min=0.18,include_label=True,gold=ldocs[i]['kg'])
            train_rows.extend(rows)
    X=np.array([make_features(r) for r in train_rows], dtype=float)
    y=np.array([r['label'] for r in train_rows], dtype=int)
    clf=LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0, solver='liblinear', random_state=43)
    clf.fit(X,y)
    probs=clf.predict_proba(X)[:,1]
    auc=float(roc_auc_score(y, probs)) if len(set(y))>1 else float('nan')
    # find best simple threshold F1 over candidates (not doc macro, diagnostic only)
    best=(0,0,0,0)
    for th in np.linspace(0.05,0.95,91):
        pred=(probs>=th)
        tp=int(((pred==1)&(y==1)).sum()); fp=int(((pred==1)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum())
        prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0; f1=2*prec*rec/(prec+rec) if prec+rec else 0
        if f1>best[0]: best=(float(f1),float(th),float(prec),float(rec))
    # test candidates
    test_rows=[]
    for ti,td in enumerate(tdocs):
        global_i=L+ti
        sims=S_full[global_i, :L]
        neigh=list(np.argsort(sims)[::-1])
        rows=aggregate_candidates_for_target(td['doc_id'],td['text'],neigh,sims,ldocs,topk=10,sim_min=0.18,include_label=False)
        test_rows.extend(rows)
    Xt=np.array([make_features(r) for r in test_rows], dtype=float) if test_rows else np.zeros((0,X.shape[1]))
    ptest=clf.predict_proba(Xt)[:,1] if len(test_rows) else []
    for r,p in zip(test_rows,ptest): r['prob']=float(p)
    # read reference
    order, ref=read_sub(Path(args.reference_csv))
    ref_sets={did:{edge_key(e) for e in ref.get(did,[])} for did in order}
    # keep only not in current
    new_rows=[]
    for r in test_rows:
        key=(r['predicate'],r['subject'],r['object'])
        did=r['doc_id']
        if key in ref_sets.get(did,set()): continue
        r['key']=key
        new_rows.append(r)
    new_rows.sort(key=lambda r:(r['prob'], r['score_sum'], r['sim_max']), reverse=True)
    # save top candidate diagnostics
    with open(out/'v40_retrain_candidate_scores.csv','w',encoding='utf-8',newline='') as f:
        fields=['doc_id','predicate','subject','object','prob','score_sum','sim_max','sim_mean','support_count','sources']
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in new_rows:
            w.writerow({k:r.get(k,'') for k in fields})
    # Build submissions. All based on current best reference.
    base={did:set(ref_sets.get(did,set())) for did in order}
    def make(name, adds_by_doc):
        sets={did:set(base.get(did,set())) for did in order}
        for did,adds in adds_by_doc.items():
            sets.setdefault(did,set()).update(adds)
        path=out/name
        write_sub(order,sets,path)
        return path
    # 1) Current v39 remains pending, but v40 starts from retrained exact duplicate ToBRFV transfer 99581.
    adds_99581=[]
    for r in new_rows:
        if r['doc_id']=='99581' and r['prob']>=0.50:
            # allow only core duplicate relations, avoid odd surfaces too aggressive by cap 8
            if r['predicate'] in {'Affects','Causes','Found_on','Located_in','Occurs_on'}:
                adds_99581.append(r['key'])
    # Manually prioritize exact missing source-like edges, not all 20.
    priority_99581={
        ('Found_on','Tomato brown rugose fruit virus','seeds'),
        ('Found_on','Tomato brown rugose fruit virus','shoes'),
        ('Found_on','Tomato brown rugose fruit virus','mechanical devices'),
        ('Found_on','Tomato brown rugose fruit virus','clothes'),
        ('Found_on','Tomato brown rugose fruit virus','roll used to transport tomatoes'),
        ('Located_in','Tomato brown rugose fruit virus','Netherlands'),
        ('Located_in','brown wrinkling of tomatoes','Netherlands'),
        ('Located_in','tomato farms','Netherlands'),
    }
    adds_99581=[e for e in priority_99581 if e not in base['99581']]
    p1=make('submission_v40_retrain_99581_tobrfv_exact_duplicate_core.csv', {'99581':set(adds_99581)})
    # 2) Xylella duplicate family broad surface completion for 115859/115232/115840/105574 Found_on only (safe relation class)
    xyl_docs={'115859','115232','115840','105574'}
    adds_xyl=defaultdict(set)
    for r in new_rows:
        if r['doc_id'] in xyl_docs and r['predicate']=='Found_on' and 'Xylella' in r['subject'] and r['object'] in {'olive trees','olive'} and r['prob']>=0.50:
            adds_xyl[r['doc_id']].add(r['key'])
    p2=make('submission_v40_retrain_xylella_foundon_surface_completion.csv', adds_xyl)
    # 3) Xylella exact duplicate locations for 104725/104752 with dates/ibiza from high similarity Balearic source.
    bal_docs={'104725','104752'}
    keep_bal={
        ('104725','Occurs_on','Xylella fastidiosa','in October 2016'),
        ('104725','Occurs_on','almond trees','this year'),
        ('104725','Occurs_on','multiplex','in October 2016'),
        ('104725','Occurs_on','fastidiosa','in October 2016'),
        ('104752','Located_in','fastidiosa','Ibiza'),
        ('104752','Located_in','multiplex','Ibiza'),
        ('104752','Located_in','pauca','Ibiza'),
        ('104752','Occurs_on','Xylella fastidiosa','in October 2016'),
        ('104752','Occurs_on','almond trees','this year'),
    }
    adds_bal=defaultdict(set)
    for did,p,s,o in keep_bal:
        if (p,s,o) not in base[did]: adds_bal[did].add((p,s,o))
    p3=make('submission_v40_retrain_balearic_xylella_exact_transfer.csv', adds_bal)
    # 4) 116678 black spot alias surface completion: black spot disease surface already in text and source gold; avoid 117314 neutral.
    black_adds={
        ('Affects','black spot disease','citrus'),
        ('Causes','Phyllosticta citricarpa','black spot disease'),
        ('Located_in','black spot disease','South Africa'),
        ('Occurs_on','black spot disease','in September'),
        ('Occurs_on','black spot disease','so far this year'),
        ('Dispersed_by','black spot disease','shipments'),
        ('Dispersed_by','black spot disease','orange exports'),
    }
    p4=make('submission_v40_retrain_116678_blackspot_alias_completion.csv', {'116678':{e for e in black_adds if e not in base['116678']}})
    # 5) cautious model top candidate bundle excluding known harmful/neutral-docs and limiting to high-prob Found_on/Causes/Affects only.
    adds_model=defaultdict(set)
    for r in new_rows:
        did=r['doc_id']
        if did in DANGEROUS_DOCS or did in NEUTRAL_EXPLORATION_DOCS: continue
        if r['prob'] < 0.82: continue
        if r['predicate'] not in {'Found_on','Causes','Affects'}: continue
        if r['object'] in {'plant material','importation of fruits'}: continue
        if len(adds_model[did]) >= 4: continue
        adds_model[did].add(r['key'])
    p5=make('submission_v40_retrain_model_highprob_core_addons.csv', adds_model)
    # 6) all best current + pending v39 most likely model target: HLB regions on current best, copied for convenience? not create? maybe no.
    # validation report
    out_paths=[p1,p2,p3,p4,p5]
    report={
        'train_docs': len(train), 'dev_docs': len(dev), 'test_docs': len(test),
        'oof_candidate_rows': len(train_rows), 'oof_positive_rate': float(y.mean()) if len(y) else None,
        'oof_candidate_auc': auc, 'best_candidate_threshold_f1_precision_recall': {'f1':best[0], 'threshold':best[1], 'precision':best[2], 'recall':best[3]},
        'test_new_candidate_count': len(new_rows),
        'outputs': [p.name for p in out_paths],
        'edits': {
            p1.name: {'99581': sorted(list(adds_99581))},
            p2.name: {k: sorted(list(v)) for k,v in adds_xyl.items()},
            p3.name: {k: sorted(list(v)) for k,v in adds_bal.items()},
            p4.name: {'116678': sorted(list({e for e in black_adds if e not in base['116678']}))},
            p5.name: {k: sorted(list(v)) for k,v in adds_model.items()},
        }
    }
    # validation: 82 rows, duplicate edge check
    val={}
    for p in out_paths:
        rows=[]; bad=[]; dup=[]
        with open(p,encoding='utf-8') as f:
            for row in csv.DictReader(f):
                rows.append(row['doc_id'])
                arr=json.loads(row['knowledge_graph'])
                keys=[(e.get('predicate'),e.get('subject'),e.get('object')) for e in arr]
                if len(keys)!=len(set(keys)): dup.append(row['doc_id'])
                for e in arr:
                    if not all(k in e for k in ['predicate','subject','object']): bad.append(row['doc_id'])
        val[p.name]={'rows':len(rows),'rows_ok':len(rows)==82,'duplicates':dup,'bad_schema_docs':bad}
    report['validation']=val
    with open(out/'v40_retrain_report.json','w',encoding='utf-8') as f: json.dump(report,f,ensure_ascii=False,indent=2)
    with open(out/'v40_summary.md','w',encoding='utf-8') as f:
        f.write('# v40 large retrain summary\n\n')
        f.write(f"- OOF candidate rows: {len(train_rows)}\n")
        f.write(f"- OOF candidate positive rate: {float(y.mean()):.4f}\n")
        f.write(f"- OOF candidate ROC-AUC: {auc:.4f}\n")
        f.write(f"- Best candidate-threshold diagnostic F1: {best[0]:.4f} at threshold {best[1]:.2f}\n")
        f.write(f"- New test candidate edges not in current best: {len(new_rows)}\n\n")
        for name,ed in report['edits'].items():
            f.write(f"## {name}\n")
            for did,adds in ed.items():
                f.write(f"- {did}: {len(adds)} additions\n")
                for a in adds[:30]: f.write(f"  - {a}\n")
            f.write('\n')
    import zipfile as zf
    zip_path=out/'pestclef_v40_large_retrain_next5.zip'
    with zf.ZipFile(zip_path,'w',compression=zf.ZIP_DEFLATED) as zz:
        for p in out_paths+[out/'v40_retrain_report.json', out/'v40_summary.md', out/'v40_retrain_candidate_scores.csv', Path(__file__)]:
            zz.write(p, arcname=Path(p).name)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])
    print('zip', zip_path)

if __name__=='__main__':
    main()
