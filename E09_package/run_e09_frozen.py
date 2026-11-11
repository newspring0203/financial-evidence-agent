#!/usr/bin/env python3
"""E09 baseline: new verified NOTE questions with the unchanged Exp08-D components.

Install in the root of financial-evidence-agent (or run with that repo on PYTHONPATH).
The legacy verified Samsung 2024 CFS corpus + cached vectors must exist locally.
Does not modify Exp08-D. Does not run `tool_qa` questions.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import httpx
from LLM_test import exp07f_parser_fix_eval as exp07f
from LLM_test import exp07a_retrieval_benchmark as base
from LLM_test import exp08d_e2e_regression as frozen
from fea.retrieval import retrieve
from validate_gold import rows_from_csv, validate

MODEL = 'qwen3:1.7b'
EMBED_MODEL = 'qwen3-embedding:4b'
EVALUATED_SCOPES = {'note_qa','multi_evidence','abstention'}
CONDITION = 'hybrid_explicit_cfs'


def sha_path(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def sha_text(x):return hashlib.sha256(x.encode('utf-8')).hexdigest()


def git_head():
    try:
        return subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
    except (OSError,subprocess.CalledProcessError):return 'unknown'


def verify_local_evidence(questions, chunks, docs):
    by_id={x['id']:x for x in chunks};by_doc={x['id']:x for x in docs}
    for g in questions:
        for e in json.loads(g['evidence_json']):
            chunk=by_id.get(e['chunk_id'])
            doc=by_doc.get(e['source_id'])
            if not chunk or not doc:
                raise ValueError(f"{g['question_id']}: Gold source/chunk missing from frozen corpus")
            if chunk['basis_context']!='CFS' or chunk['source_id']!=e['source_id']:
                raise ValueError(f"{g['question_id']}: non-CFS or source mismatch")
            start,end=e['start'],e['end']
            if not (chunk['start']<=start<end<=chunk['end']):
                raise ValueError(f"{g['question_id']}: Gold offsets outside chunk")
            if doc['text'][start:end]!=e['quote'] or sha_text(e['quote'])!=e['sha256']:
                raise ValueError(f"{g['question_id']}: Gold quote/provenance mismatch")


def run_one(client, question, docs, chunks, vectors, lookup, cfs_ids, doc_map,
            selector_predict,answer_predict):
    ts=time.perf_counter()
    bm25=[r['id'] for r in retrieve(docs,question,basis='CFS',k=len(chunks),
                                     structure=True,expansion=True,rerank=True,diversity=False)]
    qvec=base.ollama_embed(client,EMBED_MODEL,[question])[0]
    dense=base.rank_dense(chunks,vectors,qvec)
    chosen_ids=base.rrf([cid for cid in bm25 if cid in cfs_ids],
                        [cid for cid in dense if cid in cfs_ids])[:3]
    chosen=[lookup[cid] for cid in chosen_ids]
    passages=[{'ref':f'E{i}','chunk_id':c['id'],'basis_context':c['basis_context'],
               'section':c['section'],'quote':c['quote']} for i,c in enumerate(chosen,1)]
    candidates=frozen.make_candidates(question,passages)
    status='insufficient_evidence';answer='제공된 검증 가능 근거만으로 확인할 수 없습니다.'
    selection_error='';format_error='';sel_raw='';gen_raw='';selected=[]
    selection_status='';selection_reason='';selector_meta={};generator_meta={}
    if candidates:
        try:
            sel_raw,selector_meta=frozen.ollama_chat(client,MODEL,
                frozen.select_messages(question,candidates),frozen.selector_schema(candidates),selector_predict)
            sel,selection_error=frozen.parse_selection(sel_raw,{c['candidate_id'] for c in candidates})
            if not selection_error:
                selection_status=sel['status'];selection_reason=sel.get('reason','')
                selected,selection_error=frozen.verify_selection(sel,candidates,lookup,doc_map)
            if selected and not selection_error:
                gen_raw,generator_meta=frozen.ollama_generate_text(client,MODEL,
                    frozen.answer_messages(question,selected),answer_predict)
                generated,issue=frozen.parse_generation(gen_raw)
                if issue and issue!='generator_abstained':format_error=issue
                elif not issue:
                    status='answered';answer=generated['answer']
        except (ValueError,httpx.HTTPError) as ex:
            selection_error='runtime_error:'+type(ex).__name__
    else:selection_error='no_explicit_cfs_candidate_in_top3'
    # Only attach citations for successful and validated final answers.
    citations=selected if status=='answered' and not selection_error and not format_error else []
    return dict(condition=CONDITION,status=status,answer=answer,
        retrieved_ids=chosen_ids,
        selected_evidence=citations,selected_candidate_ids=[c['candidate_id'] for c in citations],
        citation_chunk_ids=[c['chunk_id'] for c in citations],citation_refs=[c['ref'] for c in citations],
        selection_status=selection_status,selection_reason=selection_reason,
        selection_error=selection_error,format_error=format_error,
        candidate_ids_offered=[c['candidate_id'] for c in candidates],
        selector_raw_response=sel_raw,generator_raw_response=gen_raw,
        elapsed_s=round(time.perf_counter()-ts,3))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gold',type=Path,required=True)
    p.add_argument('--review-dir',type=Path,required=True,help='Original Exp07-C dataset directory')
    p.add_argument('--audit-dir',type=Path,required=True,help='Original Exp07-E scope audit directory')
    p.add_argument('--cache-dir',type=Path,default=base.DEFAULT_CACHE)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--open-holdout',action='store_true',help='Deliberately consume sealed holdout once')
    p.add_argument('--selector-num-predict',type=int,default=256)
    p.add_argument('--answer-num-predict',type=int,default=350)
    p.add_argument('--timeout',type=float,default=300)
    args=p.parse_args()
    if args.out_dir.exists():p.error('out-dir already exists; never overwrite run artifacts')
    rows=rows_from_csv(args.gold)
    errors,_=validate(rows,args.gold.parent)
    if errors:p.error('Invalid Gold: '+'; '.join(errors))
    split='sealed_holdout' if args.open_holdout else 'new_eval'
    selected=[r for r in rows if r.get('review_status')=='verified' and
              r['split']==split and r['qa_scope'] in EVALUATED_SCOPES]
    unsupported=[r['question_id'] for r in rows if r.get('review_status')=='verified' and
                 r['split']==split and r['qa_scope'] not in EVALUATED_SCOPES]
    if not selected:p.error('No verified note QA/abstention/multi-evidence items in requested split')
    # Critical: the original Exp07-F loader is hardcoded to Samsung 2024 CFS.
    for r in selected:
        if r.get('company')!='삼성전자' or r.get('fiscal_year')!='2024' or r.get('basis')!='CFS':
            p.error(f"{r['question_id']}: original 08D corpus accepts only 삼성전자/2024/CFS. Other populations need separate protocol.")
    docs,old_chunks,chunks,old_digest,digest,snapshot,_,_=exp07f.verify_and_load(args)
    verify_local_evidence(selected,chunks,docs)
    frozen_config={
        'code_git_head':git_head(),
        'frozen_exp08d_sha256':sha_path(Path(frozen.__file__)),
        'frozen_selector_prompt_sha256':sha_text(frozen.SELECT_SYSTEM),
        'frozen_answer_prompt_sha256':sha_text(frozen.ANSWER_SYSTEM),
        'gold_sha256':sha_path(args.gold),'corpus_sha256':digest,
        'model':MODEL,'embed_model':EMBED_MODEL,'seed':42,'temperature':0,
        'condition':CONDITION,'top_k':3,'selector_num_predict':args.selector_num_predict,
        'answer_num_predict':args.answer_num_predict,
        'split':split,'snapshot_path':str(snapshot),'run_utc':datetime.utcnow().isoformat()+'Z'
    }
    args.out_dir.mkdir(parents=True,exist_ok=False)
    (args.out_dir/'config.json').write_text(json.dumps(frozen_config,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.out_dir/'skipped_scope.json').write_text(json.dumps(unsupported,ensure_ascii=False,indent=2),encoding='utf-8')
    docmap={d['id']:d for d in docs};lookup={c['id']:c for c in chunks}
    cfs_ids={c['id'] for c in chunks if c['basis_context']=='CFS'}
    with httpx.Client(timeout=httpx.Timeout(args.timeout,connect=15.0)) as client:
        vectors=exp07f.old_cache_vectors(EMBED_MODEL,old_chunks,old_digest,chunks,args.cache_dir)
        with (args.out_dir/'attempts.jsonl').open('w',encoding='utf-8') as out:
            for idx,g in enumerate(selected,1):
                record=run_one(client,g['question'],docs,chunks,vectors,lookup,cfs_ids,docmap,
                               args.selector_num_predict,args.answer_num_predict)
                record.update(question_id=g['question_id'],question=g['question'],topic=g['category'],
                              qa_scope=g['qa_scope'],company=g['company'],fiscal_year=g['fiscal_year'])
                gold_ids={e['chunk_id'] for e in json.loads(g['evidence_json'])}
                record['retrieval_hit_at_3']=int(bool(set(record['retrieved_ids']) & gold_ids)) if gold_ids else None
                record['citation_contains_gold']=int(bool(set(record['citation_chunk_ids']) & gold_ids)) if gold_ids else None
                out.write(json.dumps(record,ensure_ascii=False)+'\n');out.flush()
                print(f"[{idx}/{len(selected)}] {g['question_id']} {record['status']}",flush=True)
    print('Wrote actual Ollama logs to',args.out_dir)
    if unsupported:print('Separate tool_qa evaluation required:',len(unsupported),'items')

if __name__=='__main__':main()
