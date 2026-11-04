"""Exp07-F: controlled before/after retrieval comparison for a scope-parser fix.

Prerequisites: existing Exp07-C, Exp07-E scope audit, and original Exp07
701-chunk embedding caches. Keeps all original artifacts immutable.

This script verifies the old 701-chunk digest and labels, reconstructs
old cache ID->vector alignment, and scores the new 690-chunk corpus by
subsetting original vectors (same exact text and chunk IDs). It only calls
Ollama to embed 14 queries per model; never re-embeds the corpus.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import httpx

from LLM_test import exp07c_multitopic_evaluation as exp07c
from LLM_test import exp07a_retrieval_benchmark as base
from fea.retrieval import retrieve, structured_chunks

K = 3
DEFAULT_MODELS = ["qwen3-embedding:0.6b", "qwen3-embedding:4b"]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha_bytes(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_and_load(args):
    review_dir = args.review_dir.resolve()
    audit_dir = args.audit_dir.resolve()
    original = load_json(review_dir / "dataset_metadata.json")
    audit_meta = load_json(audit_dir / "metadata.json")
    if Path(audit_meta["review_dir"]).resolve() != review_dir:
        raise ValueError("07-E 감사 폴더와 07-C Gold 폴더가 일치하지 않습니다.")
    if original["corpus_sha256"] != audit_meta["corpus_sha256"]:
        raise ValueError("07-E가 확인한 원본 코퍼스 해시가 07-C와 다릅니다.")
    if sha_bytes(review_dir / "gold_review.csv") != audit_meta["gold_csv_sha256"]:
        raise ValueError("Gold CSV가 07-E 실험 준비 이후 변경됐습니다.")

    docs, new_chunks, new_digest, snapshot = exp07c.load_corpus(cache_dir=args.cache_dir)
    new_by_id = {ch["id"]: ch for ch in new_chunks}
    all_by_id = {}
    for doc in docs:
        for chunk in structured_chunks(doc):
            if chunk["id"] in all_by_id:
                raise ValueError("원문 청크 ID 중복")
            all_by_id[chunk["id"]] = chunk

    old_audit = read_csv(audit_dir / "chunk_scope_audit.csv")
    old_ids = [r["chunk_id"] for r in old_audit]
    if len(old_ids) != len(set(old_ids)) or len(old_ids) != int(original["corpus_chunks"]):
        raise ValueError("07-E 이전 검색 후보 목록의 개수 또는 ID가 맞지 않습니다.")
    if any(cid not in all_by_id for cid in old_ids):
        raise ValueError("원본 청크 ID가 현재 추출된 문서에서 사라졌습니다.")
    old_chunks = [all_by_id[cid] for cid in old_ids]
    old_digest = base.corpus_digest(old_chunks)
    if old_digest != original["corpus_sha256"]:
        raise ValueError("원본 701개 청크의 ID·원문·순서가 07-C와 달라졌습니다.")
    if base.corpus_digest(new_chunks) != new_digest:
        raise ValueError("신규 코퍼스 해시 불일치")

    # Distinguish metadata-only changes from actual chunking/text changes.
    changes = Counter((r["original_basis"], all_by_id[r["chunk_id"]]["basis_context"])
                      for r in old_audit)
    removed = set(old_ids) - set(new_by_id)
    added = set(new_by_id) - set(old_ids)
    if added:
        raise ValueError(f"새 청크가 생성되어 subset-vector 방식이 불가능합니다: {len(added)}개")
    if any(all_by_id[cid]["basis_context"] != "OFS" for cid in removed):
        raise ValueError("삭제된 기존 후보 중 명확한 OFS가 아닌 문단이 존재합니다.")

    gold_rows = read_csv(review_dir / "gold_review.csv")
    question_info = {q[0]: (q[1], q[2]) for q in exp07c.QUESTION_BANK}
    positives = defaultdict(set)
    seen = set()
    for r in gold_rows:
        qid, cid = r["question_id"], r["chunk_id"]
        if qid not in question_info or (qid, cid) in seen:
            raise ValueError(f"질문 ID 또는 중복 Gold 라벨 오류: {qid} {cid}")
        seen.add((qid, cid))
        if r["corpus_sha256"] != old_digest:
            raise ValueError("Gold 행이 원래 07-C 코퍼스를 참조하지 않습니다.")
        if r["question"] != question_info[qid][1]:
            raise ValueError(f"Gold 질문 내용 불일치: {qid}")
        chunk = all_by_id.get(cid)
        if chunk is None or r["quote"] != chunk["quote"]:
            raise ValueError(f"원문 또는 ID 변경: {cid}")
        if hashlib.sha256(chunk["quote"].encode("utf-8")).hexdigest() != r["quote_sha256"]:
            raise ValueError(f"Gold 문단 해시 불일치: {cid}")
        label = (r.get("manual_verified") or "").strip()
        if label not in ("", "0", "1"):
            raise ValueError("manual_verified는 빈칸 또는 0 또는 1이어야 합니다.")
        if label == "1":
            new_gold = new_by_id.get(cid)
            if not new_gold or new_gold["basis_context"] != "CFS":
                raise ValueError(f"확인된 Gold의 ID 또는 CFS 범위가 유지되지 않았습니다: {cid}")
            positives[qid].add(cid)
    verified = [(qid, topic, question, positives[qid])
                for qid, (topic, question) in question_info.items() if positives[qid]]

    print("[실험 무결성] 원본 07-C 코퍼스 해시 확인 / Gold CSV 및 원문 검증 완료")
    print(f"[검색 후보] 이전 {len(old_chunks)}개 → 현재 {len(new_chunks)}개 (제외 {len(removed)}개, 추가 {len(added)}개)")
    print("[범위 변경] " + ", ".join(f"{a}→{b}: {n}개" for (a, b), n in sorted(changes.items())))
    print(f"[Gold] {len(verified)}개 질문 / 라벨 {sum(len(g) for *_, g in verified)}건 / 고유 ID {len(set().union(*(g for *_, g in verified)))}개")
    print(f"[새 코퍼스 해시] {new_digest}")
    return docs, old_chunks, new_chunks, old_digest, new_digest, snapshot, verified, changes


def old_cache_vectors(model, old_chunks, old_digest, new_chunks, cache_dir):
    model_tag = hashlib.sha256(model.encode("utf-8")).hexdigest()[:12]
    cache_path = cache_dir / f"vectors_{model_tag}_{old_digest[:18]}.json.gz"
    if not cache_path.exists():
        raise FileNotFoundError(f"기존 701개 벡터 캐시가 없습니다: {cache_path}")
    with gzip.open(cache_path, "rt", encoding="utf-8") as f:
        saved = json.load(f)
    vectors = saved.get("vectors")
    if saved.get("model") != model or saved.get("corpus_sha256") != old_digest or not isinstance(vectors, list) or len(vectors) != len(old_chunks):
        raise ValueError(f"원본 벡터 캐시 무결성 검증 실패: {cache_path}")
    id_to_vec = {ch["id"]: v for ch, v in zip(old_chunks, vectors)}
    new_vectors = [base.normalize(id_to_vec[ch["id"]]) for ch in new_chunks]
    if not new_vectors or len({len(v) for v in new_vectors}) != 1:
        raise ValueError("선택된 임베딩 벡터의 차원이 서로 다릅니다.")
    print(f"[원래 벡터 재사용] {model} / 701개 캐시에서 {len(new_vectors)}개 추출")
    return new_vectors


def summary(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["method"]].append(r)
    return [{"method": name, "questions": len(rs),
             "hit_at_3": round(sum(r["hit_at_3"] for r in rs)/len(rs), 4),
             "mrr_at_3": round(sum(r["mrr_at_3"] for r in rs)/len(rs), 4),
             "recall_at_3": round(sum(r["recall_at_3"] for r in rs)/len(rs), 4)}
            for name, rs in grouped.items()]


def run(args):
    start = time.perf_counter()
    docs, old_chunks, chunks, old_digest, digest, snapshot, verified, changes = verify_and_load(args)
    baseline_summary = {r["method"]: r for r in read_csv(args.baseline_dir / "summary.csv")}
    baseline_rows = read_csv(args.baseline_dir / "per_question.csv")
    baseline_ids = {r["question_id"] for r in baseline_rows}
    if baseline_ids != {q[0] for q in verified}:
        raise ValueError("비교용 07-C 평가 질문과 현재 Gold 질문이 다릅니다.")
    baseline_meta = load_json(args.baseline_dir / "metadata.json")
    if baseline_meta["corpus_sha256"] != old_digest or sha_bytes(args.review_dir / "gold_review.csv") != baseline_meta["review_csv_sha256"]:
        raise ValueError("07-C 기준 평가의 코퍼스/Gold 스냅샷이 현재 원본과 다릅니다.")

    cfs_ids = {c["id"] for c in chunks if c["basis_context"] == "CFS"}
    lookup = {ch["id"]: ch for ch in chunks}
    rows = []
    top_rows = []
    bms = {}
    for qid, topic, question, gold in verified:
        bms[qid] = [r["id"] for r in retrieve(
            docs, question, basis="CFS", k=len(chunks),
            structure=True, expansion=True, rerank=True, diversity=False)]
        if any(cid not in lookup for cid in bms[qid]):
            raise ValueError("BM25 검색 범위와 코퍼스 목록이 불일치합니다.")

    with httpx.Client(timeout=300.0) as client:
        for model in args.models:
            vectors = old_cache_vectors(model, old_chunks, old_digest, chunks, args.cache_dir)
            for i, (qid, topic, question, gold) in enumerate(verified, 1):
                q_vec = base.ollama_embed(client, model, [question])[0]
                dense = base.rank_dense(chunks, vectors, q_vec)
                bm = bms[qid]
                hybrid = base.rrf(bm, dense)
                hybrid_cfs = base.rrf([x for x in bm if x in cfs_ids],
                                       [x for x in dense if x in cfs_ids])
                choices = [(f"Dense::{model}", dense),
                           (f"Hybrid_RRF::{model}", hybrid),
                           (f"Hybrid_CFS_only::{model}", hybrid_cfs)]
                if model == args.models[0]:
                    choices.insert(0, ("BM25", bm))
                for method, ranking in choices:
                    metric = base.metrics_at_k(ranking, gold, K)
                    rows.append({"question_id": qid, "topic": topic, "question": question,
                                 "method": method, **metric, "gold_count": len(gold)})
                    for rank, cid in enumerate(ranking[:K], 1):
                        ch = lookup[cid]
                        top_rows.append({"question_id": qid, "method": method, "rank": rank,
                                         "chunk_id": cid, "is_gold": int(cid in gold),
                                         "basis_context": ch["basis_context"],
                                         "section": ch["section"],
                                         "quote_preview": re.sub(r"\s+", " ", ch["quote"])[:220]})
                print(f"  [{i}/{len(verified)}] {qid} / {model}", flush=True)

    current = summary(rows)
    comparison = []
    print("\n=== Experiment 07-F / 파서 수정 전후 (동일 Gold 및 원문 문단) ===")
    for r in current:
        method = r["method"]
        base_r = baseline_summary.get(method)
        delta = None if base_r is None else round(r["hit_at_3"] - float(base_r["hit_at_3"]), 4)
        comparison.append({"method": method, "old_hit_at_3": base_r["hit_at_3"] if base_r else "",
                           "new_hit_at_3": r["hit_at_3"], "delta_hit_at_3": delta if delta is not None else "",
                           "old_mrr_at_3": base_r["mrr_at_3"] if base_r else "",
                           "new_mrr_at_3": r["mrr_at_3"], "old_recall_at_3": base_r["recall_at_3"] if base_r else "",
                           "new_recall_at_3": r["recall_at_3"]})
        if base_r:
            print(f"{method:<42} Hit@3 {float(base_r['hit_at_3']):.1%} → {r['hit_at_3']:.1%} "
                  f"(Δ {delta:+.1%}) | MRR {float(base_r['mrr_at_3']):.3f} → {r['mrr_at_3']:.3f}")
        else:
            print(f"{method:<42} Hit@3 {r['hit_at_3']:.1%} | MRR {r['mrr_at_3']:.3f} | Recall {r['recall_at_3']:.1%}")

    old_rank = {(r["question_id"], r["method"]): r for r in baseline_rows}
    print("\n=== 질문별 Gold 첫 등장 순위 / 수정 전 → 수정 후 ===")
    for qid in ("R07", "R14", "R16"):
        print(f"[{qid}]")
        for method in ("BM25", "Dense::qwen3-embedding:4b", "Hybrid_RRF::qwen3-embedding:4b"):
            old = old_rank.get((qid, method), {}).get("first_gold_rank") or "-"
            now = next((str(r["first_gold_rank"] or "-") for r in rows
                        if r["question_id"] == qid and r["method"] == method), "-")
            print(f"  {method:<38} {old} → {now}")

    out = args.review_dir / ("evaluation_parser_fix_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(parents=True, exist_ok=False)
    exp07c.write_csv(out / "summary.csv", current, ["method", "questions", "hit_at_3", "mrr_at_3", "recall_at_3"])
    exp07c.write_csv(out / "comparison.csv", comparison, ["method", "old_hit_at_3", "new_hit_at_3", "delta_hit_at_3", "old_mrr_at_3", "new_mrr_at_3", "old_recall_at_3", "new_recall_at_3"])
    exp07c.write_csv(out / "per_question.csv", rows, ["question_id", "topic", "question", "method", "hit_at_3", "mrr_at_3", "recall_at_3", "first_gold_rank", "gold_count"])
    exp07c.write_csv(out / "top_results.csv", top_rows, ["question_id", "method", "rank", "chunk_id", "is_gold", "basis_context", "section", "quote_preview"])
    meta = {"experiment": "exp07f_parser_fix_controlled_comparison",
            "source_snapshot": str(snapshot), "original_corpus_digest": old_digest,
            "new_corpus_digest": digest, "gold_csv_sha256": sha_bytes(args.review_dir / "gold_review.csv"),
            "original_corpus_chunks": len(old_chunks), "updated_corpus_chunks": len(chunks),
            "scope_transitions": {f"{a}->{b}": n for (a, b), n in changes.items()},
            "models": args.models, "questions": len(verified), "gold_labels": sum(len(g) for *_, g in verified),
            "original_baseline": str(args.baseline_dir), "elapsed_seconds": round(time.perf_counter()-start, 2),
            "limitations": ["Corpus vectors are reused by exact ID+text from original cached embeddings",
                            "Only query embeddings are freshly computed; Ollama configuration should remain identical",
                            "CFS/unknown metadata derives from heuristic parser; not independently validated for all passages",
                            "Strict CFS-only condition is exploratory and label-favored",
                            "Dataset small and questions authored from source; no held-out test or causal generalization",
                            "BM25 IDF remains based on all 1439 structured chunks, as in baseline; candidate scope changes"]}
    (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[결과 저장] {out}")
    print("[기존 실험/Gold/벡터 파일 변경 없음]")
    return out


def main():
    parser = argparse.ArgumentParser(description="Exp07-F: scope-parser bug fix before/after retrieval evaluation")
    parser.add_argument("--review-dir", required=True, type=Path)
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", default=base.DEFAULT_CACHE, type=Path)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, KeyError, FileNotFoundError, httpx.HTTPError, OSError, KeyboardInterrupt) as e:
        parser.exit(1, f"[실험 중단] {e}\n")


if __name__ == "__main__":
    main()
