#!/usr/bin/env python3
"""Join frozen model run logs to independently verified E09 Gold and human review.

Never infer semantic correctness from string matches, citations, hashes, or completion alone.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from validate_gold import rows_from_csv, validate

CORRECTNESS={"correct","partial","incorrect","abstained"}
RELEVANCE={"1","partial","0","NA"}
CITATION={"supported","partial","unsupported","NA"}


def load_jsonl(path):
    with open(path,encoding="utf-8") as f:return [json.loads(line) for line in f if line.strip()]


def load_reviews(path):
    rows=rows_from_csv(path);out={}
    for r in rows:
        qid=r.get("question_id", "")
        if qid in out:raise ValueError(f"duplicate review: {qid}")
        out[qid]=r
    return out


def percent(n,d):return round(n/d,4) if d else None


def group_summary(rows):
    # rows are prejoined (Gold, a single model result, optional human review)
    n=len(rows)
    run=[x for x in rows if x['attempt'] is not None]
    answerable=[x for x in run if x['gold']['answerability']=='answerable']
    model_answered=[x for x in run if x['attempt'].get('status')=='answered']
    # Exact top3 check uses verified Gold IDs, not run-produced hit flags.
    hits=[x for x in answerable if x['retrieval_hit'] is True]
    all_hits=[x for x in answerable if x['retrieval_all_hit'] is True]
    reviewed=[x for x in run if x['review'] is not None]
    corrections={c:sum(x['review'].get('answer_correctness')==c for x in reviewed) for c in CORRECTNESS}
    supported=sum(x['review'].get('citation_support')=='supported' for x in reviewed)
    selected_rel=sum(x['review'].get('selection_relevance')=='1' for x in reviewed)
    abstain_appropr=[x for x in run if x['gold']['answerability'] in ('answerable','unanswerable')]
    proper=sum((x['gold']['answerability']=='answerable')==(x['attempt'].get('status')=='answered') for x in abstain_appropr)
    return dict(n_gold=n,attempted=len(run),unrun=n-len(run),answered=len(model_answered),
        hit_at_3_n=len(hits),hit_at_3_denom=len(answerable),hit_at_3=percent(len(hits),len(answerable)),
        all_gold_hit_at_3_n=len(all_hits),all_gold_hit_at_3_denom=len(answerable),
        all_gold_hit_at_3=percent(len(all_hits),len(answerable)),
        human_reviewed=len(reviewed),unreviewed=len(run)-len(reviewed),
        complete_correct=corrections['correct'],partial=corrections['partial'],
        incorrect=corrections['incorrect'],abstained_review=corrections['abstained'],
        full_correct_rate_reviewed=percent(corrections['correct'],len(reviewed)),
        cited_supported_reviewed=supported,selection_relevant_reviewed=selected_rel,
        abstention_appropriate_n=proper,abstention_appropriate_denom=len(abstain_appropr),
        abstention_appropriate_rate=percent(proper,len(abstain_appropr)))


def normalize_ids(obj):
    if isinstance(obj,list):return [str(v).strip() for v in obj]
    if isinstance(obj,str):
        s=obj.strip()
        if s.startswith('['):
            try:
                out=json.loads(s)
                if isinstance(out,list):return normalize_ids(out)
            except json.JSONDecodeError:pass
        return [v.strip() for v in s.split('|') if v.strip()]
    return []


def compute_retrieval(gold,attempt):
    if attempt is None or gold['answerability']!='answerable':return None,None
    ids={str(e['chunk_id']) for e in json.loads(gold['evidence_json'])}
    chosen=set(normalize_ids(attempt.get('retrieved_ids'))[:3])
    if not ids:return None,None
    return bool(ids&chosen), ids<=chosen


def evaluate(gold_rows,attempt_rows,reviews,include_holdout=False,include_regression=False):
    verified=[r for r in gold_rows if r.get('review_status')=='verified']
    allowed={'new_eval'}
    if include_holdout:allowed.add('sealed_holdout')
    if include_regression:allowed.add('regression')
    verified=[r for r in verified if r.get('split') in allowed]
    attempts={}
    for r in attempt_rows:
        if r.get('condition','hybrid_explicit_cfs')!='hybrid_explicit_cfs':continue
        qid=r.get('question_id')
        if qid in attempts:raise ValueError(f"duplicate attempt for {qid} in hybrid_explicit_cfs")
        attempts[qid]=r
    joined=[]
    for g in verified:
        attempt=attempts.get(g['question_id'])
        review=reviews.get(g['question_id'])
        if not attempt or not review or not (review.get('reviewer') or '').strip():review=None
        else:
            if review.get('answer_correctness') not in CORRECTNESS:raise ValueError(f"invalid correctness for {g['question_id']}")
            if review.get('selection_relevance') not in RELEVANCE:raise ValueError(f"invalid relevance for {g['question_id']}")
            if review.get('citation_support') not in CITATION:raise ValueError(f"invalid citation for {g['question_id']}")
        hit,all_hit=compute_retrieval(g,attempt)
        joined.append({'gold':g,'attempt':attempt,'review':review,'retrieval_hit':hit,'retrieval_all_hit':all_hit})
    groups={'all':group_summary(joined)}
    for key in ('split','category','qa_scope'):
        vals=sorted({r['gold'][key] for r in joined})
        groups[key]={val:group_summary([r for r in joined if r['gold'][key]==val]) for val in vals}
    issues=[]
    for row in joined:
        if row['attempt'] is None:issues.append({'question_id':row['gold']['question_id'],'issue':'missing_attempt'})
        elif row['review'] is None:issues.append({'question_id':row['gold']['question_id'],'issue':'missing_human_review'})
    return {'note':'Only verified Gold rows are scored. Not a generalization claim without frozen real runs and human review.',
            'groups':groups,'issues':issues,'excluded_draft_gold':len(gold_rows)-sum(r.get('review_status')=='verified' for r in gold_rows)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gold',type=Path,required=True)
    p.add_argument('--attempts',type=Path,required=True)
    p.add_argument('--review',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--include-holdout',action='store_true')
    p.add_argument('--include-regression',action='store_true')
    a=p.parse_args()
    gold=rows_from_csv(a.gold)
    errors,_=validate(gold,a.gold.parent)
    if errors:raise SystemExit('Invalid Gold: '+ '\n'.join(errors))
    results=evaluate(gold,load_jsonl(a.attempts),load_reviews(a.review),a.include_holdout,a.include_regression)
    a.out.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Report:',a.out,'verified_eval_n=',results['groups']['all']['n_gold'],
        'human_reviewed=',results['groups']['all']['human_reviewed'])
    if not results['groups']['all']['n_gold']:
        print('No verified Gold rows: no AI performance figure can be reported yet.')

if __name__=='__main__':main()
