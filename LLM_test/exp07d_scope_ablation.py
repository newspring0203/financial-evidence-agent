"""Experiment 07-D: source-scope (explicit CFS) filtering ablation.

Place this file at LLM_test/exp07d_scope_ablation.py in the existing
financial-evidence-agent repository, then run:

  python -m LLM_test.exp07d_scope_ablation \
      --review-dir experiments/ollama_runs/exp07c_dataset_20261104_101436_828977

Does not alter the original questions, labels, corpus, or cached vectors.
It compares the original rankings with CFS-only *post-retrieval* filtering,
and compares Hybrid RRF scope filtering before vs. after fusion.

IMPORTANT: This is a controlled retrieval ablation, not a definitive fix.
CFS-only filtering is based solely on existing basis_context metadata. Its
accuracy needs independent audit; unknown does not imply OFS. BM25 still uses
IDF computed over the original full corpus; a true pre-index CFS filter would
require changing the corpus supplied to BM25.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import httpx

from LLM_test import exp07c_multitopic_evaluation as exp07c
from LLM_test import exp07a_retrieval_benchmark as base
from fea.retrieval import retrieve

K = 3


def is_explicit_cfs(chunk):
    return chunk.get("basis_context") == "CFS"


def scoped(ids, cfs_ids):
    return [cid for cid in ids if cid in cfs_ids]


def score_row(qid, topic, question, method, ids, gold):
    metrics = base.metrics_at_k(ids, gold, K)
    return {
        "question_id": qid,
        "topic": topic,
        "question": question,
        "method": method,
        "hit_at_3": metrics["hit_at_3"],
        "mrr_at_3": metrics["mrr_at_3"],
        "recall_at_3": metrics["recall_at_3"],
        "first_gold_rank": metrics["first_gold_rank"],
        "gold_count": len(gold),
    }


def summarize(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["method"]].append(r)
    summary = []
    for method, entries in grouped.items():
        n = len(entries)
        summary.append({
            "method": method,
            "evaluated_questions": n,
            "hit_at_3": round(sum(r["hit_at_3"] for r in entries) / n, 4),
            "mrr_at_3": round(sum(r["mrr_at_3"] for r in entries) / n, 4),
            "recall_at_3": round(sum(r["recall_at_3"] for r in entries) / n, 4),
        })
    return summary


def run(args):
    started = time.perf_counter()
    docs, chunks, digest, snapshot = exp07c.load_corpus(
        args.company, args.year, args.basis, args.cache_dir
    )
    verified = exp07c.audit_labels(args.review_dir, chunks, digest)
    if not verified:
        raise ValueError("수동 검증된 Gold 라벨이 없습니다.")

    by_id = {c["id"]: c for c in chunks}
    cfs_ids = {cid for cid, c in by_id.items() if is_explicit_cfs(c)}
    unknown_ids = {cid for cid, c in by_id.items() if c.get("basis_context") == "unknown"}
    print(
        f"[청크 범위] 전체 {len(chunks)}개, 명시적 CFS {len(cfs_ids)}개, "
        f"unknown {len(unknown_ids)}개, 그 외 {len(chunks)-len(cfs_ids)-len(unknown_ids)}개"
    )
    print("[원칙] unknown은 OFS로 단정하지 않음. CFS-only는 메타데이터 기반 실험적 후처리.")

    rows = []
    top_rows = []
    rank_detail = defaultdict(dict)

    # BM25 ranking is computed against the SAME original 701-chunk corpus.
    bm25_by_q = {}
    for qid, topic, question, gold in verified:
        full_bm25 = [r["id"] for r in retrieve(
            docs, question, basis=args.basis, k=len(chunks), structure=True,
            expansion=True, rerank=True, diversity=False
        )]
        bm25_by_q[qid] = full_bm25
        for method, ids in [
            ("BM25/original", full_bm25),
            ("BM25/CFS-post-filter", scoped(full_bm25, cfs_ids)),
        ]:
            rows.append(score_row(qid, topic, question, method, ids, gold))
            rank_detail[qid][method] = ids

    with httpx.Client(timeout=300.0) as client:
        for model in args.models:
            print(f"[모델] {model}", flush=True)
            vectors, embedded_digest, reused = base.embed_corpus(
                client, model, chunks, args.batch_size, Path(args.cache_dir)
            )
            if embedded_digest != digest:
                raise ValueError("임베딩 캐시의 문서 버전이 기존 Gold와 일치하지 않습니다.")
            print(f"[임베딩] {'기존 캐시 재사용' if reused else '새 벡터 생성 및 캐시 저장'}", flush=True)

            for i, (qid, topic, question, gold) in enumerate(verified, 1):
                qv = base.ollama_embed(client, model, [question])[0]
                dense = base.rank_dense(chunks, vectors, qv)
                bm25 = bm25_by_q[qid]
                dense_cfs = scoped(dense, cfs_ids)
                bm25_cfs = scoped(bm25, cfs_ids)
                fused_original = base.rrf(bm25, dense)
                fused_before = base.rrf(bm25_cfs, dense_cfs)
                fused_after = scoped(fused_original, cfs_ids)

                choices = [
                    (f"Dense::{model}/original", dense),
                    (f"Dense::{model}/CFS-post-filter", dense_cfs),
                    (f"Hybrid::{model}/original", fused_original),
                    (f"Hybrid::{model}/CFS-before-RRF", fused_before),
                    (f"Hybrid::{model}/CFS-after-RRF", fused_after),
                ]
                for method, ids in choices:
                    rows.append(score_row(qid, topic, question, method, ids, gold))
                    rank_detail[qid][method] = ids
                print(f"  [{i:2}/{len(verified)}] {qid}", flush=True)

    for qid, topic, question, gold in verified:
        for method, ids in rank_detail[qid].items():
            for idx, cid in enumerate(ids[:K], 1):
                c = by_id.get(cid)
                if c is None:
                    raise ValueError(f"검색 결과 ID가 코퍼스에 없습니다: {cid}")
                top_rows.append({
                    "question_id": qid,
                    "method": method,
                    "rank": idx,
                    "chunk_id": cid,
                    "is_gold": int(cid in gold),
                    "basis_context": c.get("basis_context", "unknown"),
                    "section": c.get("section", ""),
                    "quote_preview": " ".join(c.get("quote", "").split())[:250],
                })

    summary = summarize(rows)
    out = Path(args.review_dir) / (
        "evaluation_scope_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    out.mkdir(parents=True, exist_ok=False)
    exp07c.write_csv(out / "summary.csv", summary,
                     ["method", "evaluated_questions", "hit_at_3", "mrr_at_3", "recall_at_3"])
    exp07c.write_csv(out / "per_question.csv", rows,
                     ["question_id", "topic", "question", "method", "hit_at_3",
                      "mrr_at_3", "recall_at_3", "first_gold_rank", "gold_count"])
    exp07c.write_csv(out / "top_results.csv", top_rows,
                     ["question_id", "method", "rank", "chunk_id", "is_gold",
                      "basis_context", "section", "quote_preview"])

    gold_path = Path(args.review_dir) / "gold_review.csv"
    metadata = {
        "experiment": "exp07d_CFS_scope_post_retrieval_ablation",
        "source_snapshot": str(snapshot),
        "corpus_sha256": digest,
        "labels_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
        "questions": len(verified),
        "corpus_chunks": len(chunks),
        "explicit_CFS_chunks": len(cfs_ids),
        "unknown_chunks": len(unknown_ids),
        "k": K,
        "models": args.models,
        "time_seconds": round(time.perf_counter() - started, 2),
        "experimental_limits": [
            "Gold includes only manually approved explicitly labeled CFS chunks",
            "CFS-only uses *existing metadata*, which can itself be incomplete or wrong",
            "Unknown is unverified; not proof of OFS",
            "BM25 original corpus used for IDF, then ranked list is filtered",
            "Pre-RRF CFS filtering and post-RRF CFS filtering test different fusion behavior",
            "Same questions, gold IDs, corpus and embedding vectors as Experiment 07-C",
            "Exact chunk ID evaluation; unlabelled synonymous passages may appear as misses",
        ],
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== Experiment 07-D / Scope Ablation ===")
    for r in summary:
        print(f"{r['method']:<48} Hit@3={r['hit_at_3']:.1%} "
              f"MRR@3={r['mrr_at_3']:.3f} Recall@3={r['recall_at_3']:.1%}")

    print("\n=== R14/R16 gold 순위 변경 ===")
    for qid in ("R14", "R16"):
        items = [r for r in rows if r["question_id"] == qid]
        if not items:
            continue
        print(f"[{qid}]")
        for r in items:
            print(f"  {r['method']:<48} first_gold_rank={r['first_gold_rank'] or '-'} "
                  f"hit={r['hit_at_3']}")

    print(f"\n[결과 저장] {out}")
    print("[주의] 개선된 지표가 나오더라도 실제 범위 파서의 정확성 검증은 별개입니다.")
    return out


def main():
    parser = argparse.ArgumentParser(description="Exp07-D: CFS-only retrieval and RRF filtering ablation")
    parser.add_argument("--review-dir", required=True, type=Path)
    parser.add_argument("--company", default="삼성전자")
    parser.add_argument("--year", default=2024, type=int)
    parser.add_argument("--basis", default="CFS", choices=["CFS"])
    parser.add_argument("--cache-dir", default=base.DEFAULT_CACHE, type=Path)
    parser.add_argument("--models", nargs="+", default=["qwen3-embedding:0.6b", "qwen3-embedding:4b"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, FileNotFoundError, httpx.HTTPError, KeyboardInterrupt) as exc:
        parser.exit(1, f"[실험 중단] {exc}\n")


if __name__ == "__main__":
    main()
