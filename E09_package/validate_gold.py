#!/usr/bin/env python3
"""Structural/source-provenance checks for E09 Gold annotations. Standard library only.

IMPORTANT: Mechanical checks cannot establish semantic answer correctness.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

CATEGORIES={"financial_lookup","change_comparison","accounting_policy","financial_instruments","multi_evidence","abstention"}
SPLITS={"new_eval","sealed_holdout","regression"}
SCOPES={"note_qa","tool_qa","multi_evidence","abstention"}
BASIS={"CFS","OFS"}


def rows_from_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f: return list(csv.DictReader(f))


def digest(text): return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate(rows, base: Path, require_verified=False):
    errors=[];warnings=[];seen=set();counts=Counter()
    for number,r in enumerate(rows,2):
        qid=(r.get("question_id") or "").strip()
        label=f"line {number} ({qid or '?'}): "
        if not qid: errors.append(label+"question_id missing")
        if qid in seen: errors.append(label+"duplicate question_id")
        seen.add(qid)
        if r.get("category") not in CATEGORIES: errors.append(label+"invalid category")
        if r.get("split") not in SPLITS: errors.append(label+"invalid split")
        if r.get("qa_scope") not in SCOPES: errors.append(label+"invalid qa_scope")
        if r.get("basis") not in BASIS: errors.append(label+"invalid basis")
        if not (r.get("question") or "").strip(): errors.append(label+"question is empty")
        counts[(r.get("category"),r.get("split"))]+=1
        status=r.get("review_status")
        if status!="verified":
            if require_verified: errors.append(label+"Gold is not verified")
            else: warnings.append(label+"draft: not eligible for scoring")
            continue
        for field in ("company","fiscal_year","filing_id","expected_answer","gold_verified_by","verified_at"):
            if not (r.get(field) or "").strip(): errors.append(label+f"{field} missing")
        try: claims=json.loads(r.get("required_claims_json") or "[]")
        except json.JSONDecodeError: errors.append(label+"invalid claims JSON");claims=[]
        if not isinstance(claims,list):errors.append(label+"claims must be an array")
        try: evidence=json.loads(r.get("evidence_json") or "[]")
        except json.JSONDecodeError: errors.append(label+"invalid evidence JSON");evidence=[]
        if not isinstance(evidence,list): errors.append(label+"evidence must be an array");evidence=[]
        answerability=r.get("answerability")
        if answerability not in ("answerable","unanswerable"):errors.append(label+"answerability must be answerable/unanswerable")
        if answerability=="answerable":
            if not claims: errors.append(label+"required_claims_json empty on answerable item")
            if not evidence: errors.append(label+"verified answerable item needs evidence")
        elif answerability=="unanswerable":
            if not (r.get("negative_scope_review") or "").strip():
                errors.append(label+"negative_scope_review missing on unanswerable item")
        for idx,e in enumerate(evidence):
            etag=f"{label}evidence[{idx}]: "
            if not isinstance(e,dict): errors.append(etag+"must be object");continue
            for field in ("chunk_id","source_id","quote","sha256","start","end"):
                if e.get(field) in (None,""): errors.append(etag+f"missing {field}")
            quote=e.get("quote")
            if isinstance(quote,str):
                actual=digest(quote)
                if e.get("sha256")!=actual: errors.append(etag+"quote SHA-256 mismatch")
            if isinstance(e.get("start"),int) and isinstance(e.get("end"),int):
                if e["start"]<0 or e["end"]<=e["start"]:errors.append(etag+"invalid offsets")
                if isinstance(quote,str) and (e["end"]-e["start"]!=len(quote)):
                    errors.append(etag+"offset length != quote char count")
                source_file=e.get("source_text_file")
                if source_file:
                    source_path=(base/source_file).resolve()
                    if not source_path.is_relative_to(base.resolve()):errors.append(etag+"source path outside dataset directory")
                    elif not source_path.is_file():errors.append(etag+"source_text_file missing")
                    elif source_path.read_text(encoding="utf-8")[e["start"]:e["end"]]!=quote:
                        errors.append(etag+"source text/offset mismatch")
            else:errors.append(etag+"start/end must be integer")
    if len(rows)==48:
        for c in CATEGORIES:
            if counts[(c,"new_eval")]!=6 or counts[(c,"sealed_holdout")]!=2:
                errors.append(f"category {c} must have 6 new_eval + 2 holdout questions")
    return errors,warnings


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--gold",type=Path,required=True)
    p.add_argument("--require-verified",action="store_true")
    args=p.parse_args()
    rows=rows_from_csv(args.gold)
    errors,warnings=validate(rows,args.gold.parent,args.require_verified)
    print(f"rows={len(rows)} verified={sum(r.get('review_status')=='verified' for r in rows)} warnings={len(warnings)} errors={len(errors)}")
    for item in errors:print("ERROR",item)
    if warnings:print(f"NOTE {len(warnings)} draft annotations excluded from scoring")
    raise SystemExit(1 if errors else 0)

if __name__=="__main__":main()
