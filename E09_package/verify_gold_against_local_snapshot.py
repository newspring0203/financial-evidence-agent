#!/usr/bin/env python3
"""Check E09 AI-draft evidence against the user's unmodified local public filing snapshot.

Only verifies exact bytes, offsets, chunk identity, CFS scope, and SHA-256.
It does NOT verify semantic answer correctness; human approval remains mandatory.
Run from the financial-evidence-agent repository root:
    PYTHONPATH="$(pwd)" python3 E09_package/verify_gold_against_local_snapshot.py \
      --gold E09_package/gold_review_ai_draft_48.csv \
      --snapshot experiments/ollama_runs/exp07_cache/opendart_2024_CFS_73b47d6729b4.json.gz
"""
import argparse
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

from fea.retrieval import structured_chunks


def validate(gold: Path, snapshot: Path):
    with gzip.open(snapshot, 'rt', encoding='utf-8') as f:
        stored=json.load(f)
    docs=stored.get('documents',[])
    doc_by_id={d['id']:d for d in docs}
    chunk_by_id={c['id']:c for d in docs for c in structured_chunks(d)}
    errors=[]
    nrows=0
    nevidences=0
    for r in csv.DictReader(gold.open(encoding='utf-8-sig', newline='')):
        if r['split']!='new_eval':
            continue  # Do not inspect or print sealed holdout material.
        if r['qa_scope']=='tool_qa':
            continue  # Outside this note-RAG source snapshot review.
        nrows+=1
        try:
            evidence=json.loads(r.get('evidence_json') or '[]')
        except json.JSONDecodeError:
            errors.append(r['question_id']+': invalid evidence_json')
            continue
        for e in evidence:
            nevidences+=1
            qid=r['question_id']
            ch=chunk_by_id.get(e.get('chunk_id'))
            doc=doc_by_id.get(e.get('source_id'))
            if not ch or not doc:
                errors.append(qid+': missing document or chunk: '+str(e.get('chunk_id')))
                continue
            try:
                start,end=int(e['start']),int(e['end'])
                quote=e['quote']
                if ch['source_id'] != doc['id'] or ch['basis_context'] != 'CFS':
                    raise ValueError('source/CFS mismatch')
                if not (ch['start']<=start<end<=ch['end']):
                    raise ValueError('offsets outside chunk')
                if doc['text'][start:end]!=quote:
                    raise ValueError('source offsets differ')
                if hashlib.sha256(quote.encode('utf-8')).hexdigest()!=e['sha256']:
                    raise ValueError('SHA-256 mismatch')
                if ch['quote']!=quote or ch['start']!=start or ch['end']!=end:
                    raise ValueError('not the frozen full chunk')
            except (KeyError,TypeError,ValueError) as exc:
                errors.append(f'{qid}: {e.get("chunk_id")}: {exc}')
    print(f'NOTE E09 questions inspected={nrows}; proposed evidence passages checked={nevidences}; source errors={len(errors)}')
    for e in errors: print('ERROR',e)
    print('Mechanical provenance only. Human semantic review is still required.')
    return len(errors)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--gold',type=Path,required=True)
    p.add_argument('--snapshot',type=Path,required=True)
    args=p.parse_args()
    sys.exit(1 if validate(args.gold,args.snapshot) else 0)
