#!/usr/bin/env python3
"""E09 human semantic review. Explicitly human-confirmed items only.

This program does NOT decide semantic correctness. It displays every claim and
its complete literal source excerpts so the reviewer can independently check.
It never unlocks sealed holdout data, never changes the draft input, and saves
approvals as a separate file after every decision.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import sys

# Manual-review candidates, not automatic verified Gold labels.
DIRECT_CANDIDATES = (
    "E09-17", "E09-27", "E09-28", "E09-29", "E09-30",
    "E09-34", "E09-35", "E09-36", "E09-37", "E09-38",
)


def read_csv(p: Path):
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("No CSV header")
        return reader.fieldnames, list(reader)


def write_csv_atomic(p: Path, fields: list[str], rows: list[dict]):
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(p)


def check_source(row, doc_map, chunk_map):
    if row.get("split") != "new_eval" or row.get("qa_scope") not in ("note_qa", "multi_evidence"):
        raise ValueError(f"{row['question_id']}: ineligible scope/split")
    if row.get("answerability") != "answerable":
        raise ValueError(f"{row['question_id']}: not marked answerable in draft")
    if not row.get("expected_answer", "").strip():
        raise ValueError(f"{row['question_id']}: draft answer missing")
    claims = json.loads(row.get("required_claims_json") or "[]")
    ev = json.loads(row.get("evidence_json") or "[]")
    if not isinstance(claims, list) or not claims or not isinstance(ev, list) or not ev:
        raise ValueError(f"{row['question_id']}: missing claims/evidence")
    for j, e in enumerate(ev, 1):
        doc = doc_map.get(e.get("source_id"))
        ch = chunk_map.get(e.get("chunk_id"))
        if doc is None or ch is None:
            raise ValueError(f"{row['question_id']} evidence {j}: source/chunk not found")
        start, end, quote = e["start"], e["end"], e["quote"]
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(doc["text"])):
            raise ValueError(f"{row['question_id']} evidence {j}: invalid offsets")
        if not (ch["source_id"] == e["source_id"] and ch["basis_context"] == "CFS"
                and ch["start"] <= start < end <= ch["end"]):
            raise ValueError(f"{row['question_id']} evidence {j}: chunk scope mismatch")
        if doc["text"][start:end] != quote or ch["quote"][start-ch["start"]:end-ch["start"]] != quote:
            raise ValueError(f"{row['question_id']} evidence {j}: exact source quote mismatch")
        if hashlib.sha256(quote.encode("utf-8")).hexdigest() != e["sha256"]:
            raise ValueError(f"{row['question_id']} evidence {j}: SHA-256 mismatch")
    return claims, ev


def main():
    ap = argparse.ArgumentParser(description="Review E09 candidates manually; only explicit approvals become verified")
    ap.add_argument("--gold", required=True, type=Path, help="Original AI draft Gold (never modified)")
    ap.add_argument("--snapshot", required=True, type=Path, help="Original exp07 OpenDART snapshot")
    ap.add_argument("--out", required=True, type=Path, help="Separate output file for approvals")
    ap.add_argument("--reviewer", default="", help="Real human reviewer name or identifier")
    args = ap.parse_args()
    if args.out.resolve() == args.gold.resolve():
        ap.error("Output must differ from original draft")
    try:
        from fea.retrieval import structured_chunks
    except ImportError as exc:
        ap.error(f"Run from the project root with PYTHONPATH=$(pwd): {exc}")
    with gzip.open(args.snapshot, "rt", encoding="utf-8") as f:
        data = json.load(f)
    docs = data.get("documents", [])
    if not docs:
        ap.error("Original snapshot has no documents")
    doc_map = {d["id"]: d for d in docs}
    chunk_map = {ch["id"]: ch for d in docs for ch in structured_chunks(d)}
    fields, original = read_csv(args.gold)
    if args.out.exists():
        existing_fields, rows = read_csv(args.out)
        if existing_fields != fields:
            ap.error("Existing review output does not match CSV schema")
        if {r['question_id'] for r in rows} != {r['question_id'] for r in original}:
            ap.error("Existing review output question set does not match draft")
        # Reviewed data remains immutable unless this interactive tool changes that row.
    else:
        rows = original
    by_id = {r["question_id"]: r for r in rows}
    if any(qid not in by_id for qid in DIRECT_CANDIDATES):
        ap.error("Missing direct-candidate IDs")
    reviewer = (args.reviewer or input("실제 검수자 이름 또는 식별자: ")).strip()
    if not reviewer:
        ap.error("Reviewer identifier is mandatory")
    print("\nE09 원문 의미 검수. 자동 정답 판정이 아닙니다.")
    print("반드시 원문 전체, 필수 주장, 모범답변을 확인해야 합니다.")
    print("승인: APPROVE E09-17  | 건너뛰기: s | 종료: q")
    print("봉인 홀드아웃 / 수치조회 / 근거 부족 문항은 건드리지 않습니다.\n")
    for qid in DIRECT_CANDIDATES:
        r = by_id[qid]
        if r["review_status"] == "verified":
            print(f"[{qid}] 이전에 승인됨 — 건너뜀")
            continue
        try:
            claims, evs = check_source(r, doc_map, chunk_map)
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            print(f"[{qid}] 원문 무결성/Gold 구조 오류: {e}; 승인 불가")
            continue
        print("\n" + "=" * 75)
        print(qid, r["category"], "|", r["question"])
        print("\n[모범답변 초안]\n", r["expected_answer"])
        print("\n[정답으로 인정할 필수 주장]")
        for i,c in enumerate(claims,1):print(f" {i}. {c}")
        for i,e in enumerate(evs,1):
            print(f"\n[공시 원문 {i}/{len(evs)}] {e['chunk_id']} / {e['start']}:{e['end']}")
            print(e["quote"])
            print("[출처 해시 검증 통과]",e["sha256"])
        print("\n검수 질문: 모든 필수 주장을 해당 원문이 직접 뒷받침하나요?")
        print("연도·CFS·측정단위·부정/예외 조건이 맞나요? 표일 경우 행/열 위치가 맞나요?")
        answer = input(f"승인하려면 'APPROVE {qid}' 정확히 입력 (s/q): ").strip()
        if answer.lower() == "q":break
        if answer == f"APPROVE {qid}":
            r["review_status"] = "verified"
            r["gold_verified_by"] = reviewer
            r["verified_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            r["notes"] = (r.get("notes", "") + " | human semantic review approved after full quote/claims inspection").strip()
            print("승인 기록됨.")
        else:
            print("미승인 상태 유지.")
        write_csv_atomic(args.out, fields, rows)
        print("저장됨:", args.out)
    if not args.out.exists():write_csv_atomic(args.out, fields, rows)
    approved=[r["question_id"] for r in rows if r["review_status"] == "verified" and r["split"] == "new_eval"]
    print("\n최종 새 평가 검수 승인:",len(approved),"개", ", ".join(approved) or "없음")
    print("검토 파일:", args.out)
    return 0

if __name__ == "__main__":sys.exit(main())
