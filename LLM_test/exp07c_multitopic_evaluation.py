"""Exp07-C: multi-topic, manually verified OpenDART retrieval benchmark.

Run from ~/financial-evidence-agent with the original exp07 module installed:

  python -m LLM_test.exp07c_multitopic --prepare
  python -m LLM_test.exp07c_multitopic --review --review-dir experiments/ollama_runs/exp07c_dataset_...
  # Alternatively edit gold_review.csv manually; mark only valid CFS evidence.
  python -m LLM_test.exp07c_multitopic --audit --review-dir experiments/ollama_runs/exp07c_dataset_...
  python -m LLM_test.exp07c_multitopic --evaluate --review-dir experiments/ollama_runs/exp07c_dataset_...

Source content must be manually reviewed: candidates are phrase-match proposals,
NOT independently verified annotations. No answer-generation LLM is used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import httpx

from LLM_test import exp07a_retrieval_benchmark as base
from fea.retrieval import retrieve

ROOT = base.ROOT
OUTPUT_ROOT = base.OUTPUT_ROOT
CACHE_DIR = base.DEFAULT_CACHE
K = 3

# Exploratory QUESTIONS ONLY: they do not assert that the source contains an answer.
# Gold evidence must be selected from the actual report and manually verified.
QUESTION_BANK = [
    ("R01", "수익인식", "연결회사는 제품 판매 수익을 어떤 시점에 인식하나요?", ["수익인식", "통제", "고객", "인식"]),
    ("R02", "수익인식", "고객의 제품 반품이 예상될 때 어떤 자산과 부채를 인식하나요?", ["반품", "환불부채", "회수할", "제품"]),
    ("R03", "수익인식", "계약의 수행의무는 어떤 시점에 충족된 것으로 보나요?", ["수행의무", "충족", "통제", "수익"]),
    ("R04", "금융상품", "금융자산의 분류와 측정에 어떤 기준을 적용하나요?", ["금융자산", "사업모형", "계약상", "현금흐름"]),
    ("R05", "금융상품", "금융자산의 기대신용손실과 손실충당금은 어떻게 평가하나요?", ["기대신용손실", "손실충당금", "금융자산", "신용"]),
    ("R06", "특수관계자", "연결회사가 특수관계자와 수행한 주요 거래는 어느 주석에서 확인하나요?", ["특수관계자", "거래", "내역"]),
    ("R07", "특수관계자", "특수관계자 관련 채권과 채무는 어떻게 공시되어 있나요?", ["특수관계자", "채권", "채무", "잔액"]),
    ("R08", "재무위험", "연결회사는 유동성위험을 어떻게 관리하나요?", ["유동성위험", "관리", "현금"]),
    ("R09", "재무위험", "금융부채의 만기 분석에 어떤 현금흐름 정보를 사용하나요?", ["금융부채", "만기", "계약상", "현금흐름"]),
    ("R10", "재무위험", "현금흐름위험회피회계는 어떤 위험을 대상으로 적용하나요?", ["현금흐름위험회피", "위험회피", "파생상품", "재고자산"]),
    ("R11", "충당부채", "연결회사는 충당부채를 어떤 조건에서 인식하나요?", ["충당부채", "현재의무", "경제적", "유출"]),
    ("R12", "유형자산", "유형자산 감가상각에 적용하는 방법과 내용연수는 무엇인가요?", ["유형자산", "감가상각", "내용연수"]),
    ("R13", "리스", "리스부채와 사용권자산은 최초에 어떤 방식으로 측정하나요?", ["리스부채", "사용권자산", "최초", "측정"]),
    ("R14", "법인세", "이연법인세자산의 회수가능성은 어떤 기준으로 판단하나요?", ["이연법인세자산", "과세소득", "회수", "인식"]),
    ("R15", "영업부문", "연결회사는 영업부문을 어떤 기준으로 구분하나요?", ["영업부문", "최고영업의사결정자", "보고부문", "구분"]),
    ("R16", "현금", "현금및현금성자산에 포함되는 항목과 기준은 무엇인가요?", ["현금및현금성자산", "예금", "단기", "현금"]),
]

FIELDS = ["question_id", "topic", "question", "candidate_rank", "chunk_id", "section",
          "basis_context", "source_url", "quote", "quote_sha256", "corpus_sha256",
          "matched_keywords", "manual_verified", "review_notes"]


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_corpus(company="삼성전자", year=2024, basis="CFS", cache_dir=CACHE_DIR):
    if (company, year, basis) != ("삼성전자", 2024, "CFS"):
        raise ValueError("현재 평가용 질문셋은 삼성전자 2024년 연결 공시 전용입니다.")
    # The snapshot was created by original Exp07. Never collect another report silently.
    path = base.snapshot_path(Path(cache_dir), company, year, basis)
    if not path.exists():
        raise FileNotFoundError(f"기존 OpenDART 스냅샷이 없습니다: {path}\n먼저 기존 Experiment 07 --live를 실행해 주세요.")
    import gzip
    with gzip.open(path, "rt", encoding="utf-8") as f:
        saved = json.load(f)
    docs = saved.get("documents", [])
    if not docs:
        raise ValueError("공시 스냅샷의 문서 목록이 비어 있습니다.")
    chunks = base.relevant_corpus(docs, basis)
    if not chunks:
        raise ValueError("검색용 문단이 없습니다.")
    return docs, chunks, base.corpus_digest(chunks), path


def candidate_score(chunk, keywords):
    """Lexical discovery only; NOT a gold label or the evaluated retrieval metric."""
    quoted = re.sub(r"\s+", "", chunk["quote"]).lower()
    heading = re.sub(r"\s+", "", chunk.get("section", "")).lower()
    matches = [word for word in keywords if re.sub(r"\s+", "", word).lower() in quoted]
    if not matches:
        return 0, []
    score = sum((4 if len(w) >= 5 else 2) for w in matches)
    score += sum(3 for w in keywords if w in heading)
    score += 2 if chunk.get("basis_context") == "CFS" else 0
    # Require multiple signals to prevent irrelevant single-word matches.
    if len(matches) < 2:
        return 0, []
    return score, matches


def build_candidates(chunks, digest, max_candidates=5):
    rows = []
    for qid, topic, question, keywords in QUESTION_BANK:
        ranked = []
        for chunk in chunks:
            if chunk.get("basis_context") != "CFS":
                # Unknown might actually be CFS, but is not safely labeled as CFS.
                # Keep it out until an independent scope-verification mechanism exists.
                continue
            score, matches = candidate_score(chunk, keywords)
            if score:
                ranked.append((score, chunk, matches))
        ranked.sort(key=lambda x: (-x[0], x[1]["id"]))
        # Prefer distinct evidence locations, no overlapping/near-duplicate fragments.
        selected = []
        for score, chunk, matches in ranked:
            overlapping = any(c["source_id"] == chunk["source_id"] and
                              max(0, min(c["end"], chunk["end"]) - max(c["start"], chunk["start"])) > 0.45 *
                              min(c["end"] - c["start"], chunk["end"] - chunk["start"])
                              for c in selected)
            if overlapping:
                continue
            selected.append(chunk)
            rows.append({"question_id": qid, "topic": topic, "question": question,
                         "candidate_rank": len(selected), "chunk_id": chunk["id"],
                         "section": chunk.get("section", ""),
                         "basis_context": chunk.get("basis_context", "unknown"),
                         "source_url": chunk.get("url", ""), "quote": chunk["quote"],
                         "quote_sha256": hashlib.sha256(chunk["quote"].encode()).hexdigest(),
                         "corpus_sha256": digest, "matched_keywords": " | ".join(matches),
                         "manual_verified": "", "review_notes": ""})
            if len(selected) == max_candidates:
                break
    return rows


def prepare(args):
    _, chunks, digest, snapshot = load_corpus(args.company, args.year, args.basis, args.cache_dir)
    out = OUTPUT_ROOT / ("exp07c_dataset_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(parents=True, exist_ok=False)
    candidates = build_candidates(chunks, digest, args.candidates)
    write_csv(out / "gold_review.csv", candidates, FIELDS)
    (out / "question_bank.json").write_text(json.dumps([
        {"question_id": qid, "topic": topic, "question": question, "discovery_terms": terms}
        for qid, topic, question, terms in QUESTION_BANK
    ], ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "dataset_metadata.json").write_text(json.dumps({
        "source": "OpenDART original public report snapshot",
        "snapshot": str(snapshot.relative_to(ROOT)) if snapshot.is_relative_to(ROOT) else str(snapshot),
        "company": args.company, "year": args.year, "basis": args.basis,
        "corpus_chunks": len(chunks), "corpus_sha256": digest,
        "questions": len(QUESTION_BANK), "candidates": len(candidates),
        "label_policy": "Unverified until manual_verified=1 after source, question, scope and accounting-value checks",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[데이터] 삼성전자 2024 연결 공시 / 검색 문단 {len(chunks)}개")
    print(f"[평가 질문] {len(QUESTION_BANK)}개 / 후보 {len(candidates)}개 (아직 정답 검증 전)")
    for qid, topic, _, _ in QUESTION_BANK:
        print(f"  {qid} {topic:7s} 후보 {sum(r['question_id']==qid for r in candidates)}개")
    print(f"[검토 파일] {out / 'gold_review.csv'}")
    print("[중요] 직접 원문과 대조한 CFS 정답 문단에만 manual_verified=1을 입력하세요. 부정확한 후보는 0 또는 빈칸으로 두세요.")
    return out


def audit_labels(review_dir, chunks, digest):
    review_dir = Path(review_dir)
    path = review_dir / "gold_review.csv"
    if not path.exists():
        raise FileNotFoundError(f"검토 파일을 찾을 수 없습니다: {path}")
    labels = read_csv(path)
    lookup = {c["id"]: c for c in chunks}
    if not labels:
        raise ValueError("정답 후보 행이 없습니다.")
    checked = set()
    picked = defaultdict(set)
    for row in labels:
        qid = row.get("question_id")
        cid = row.get("chunk_id")
        if qid not in {q[0] for q in QUESTION_BANK}:
            raise ValueError(f"등록되지 않은 question_id: {qid}")
        if (qid, cid) in checked:
            raise ValueError(f"중복 질문/문단 라벨: {qid} / {cid}")
        checked.add((qid, cid))
        if row.get("corpus_sha256") != digest:
            raise ValueError("실험 문서가 바뀌었습니다. 기존 정답 라벨을 그대로 재사용할 수 없습니다.")
        chunk = lookup.get(cid)
        if not chunk or chunk["quote"] != row.get("quote") or \
                hashlib.sha256(chunk["quote"].encode()).hexdigest() != row.get("quote_sha256"):
            raise ValueError(f"원문 내용이나 청크 ID가 바뀌었습니다: {qid} {cid}")
        label = (row.get("manual_verified") or "").strip()
        if label not in ("", "0", "1"):
            raise ValueError(f"manual_verified에는 1, 0 또는 빈칸만 입력할 수 있습니다: {qid}")
        if label == "1":
            if chunk.get("basis_context") != "CFS":
                raise ValueError(f"연결 맥락을 확인하지 못한 문단은 채점에서 제외합니다: {qid} {cid}")
            picked[qid].add(cid)
    verified = [(qid, topic, question, picked[qid]) for qid,topic,question,_ in QUESTION_BANK if picked[qid]]
    verified_topics = {topic for _,topic,_,_ in verified}
    distinct_gold = set().union(*(gold for *_,gold in verified)) if verified else set()
    print(f"[검토] {len(labels)}개 후보 중 정답 라벨 {sum(len(v) for v in picked.values())}건")
    print(f"[평가 가능] {len(verified)}/{len(QUESTION_BANK)}개 질문 · {len(verified_topics)}개 주제 · 고유 정답 문단 {len(distinct_gold)}개")
    for qid, topic, question, gold in verified:
        print(f"  [확인] {qid} {topic:7s} 정답 문단 {len(gold)}개")
    missing = [q[0] for q in QUESTION_BANK if not picked[q[0]]]
    if missing:
        print("[미검토/정답없음] " + ", ".join(missing))
    if len(verified_topics) < 4 or len(distinct_gold) < 4:
        print("[주의] 서로 다른 주석 주제/정답 문단이 아직 충분히 다양하지 않습니다. 결과는 탐색적 파일럿입니다.")
    return verified


def review_interactively(review_dir):
    """Prompt per question. Selected ranks are manual evidence choices, not predicted labels."""
    review_dir = Path(review_dir)
    path = review_dir / "gold_review.csv"
    rows = read_csv(path)
    if not rows:
        raise ValueError("검토할 정답 후보가 없습니다.")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["question_id"]].append(row)
    print("[검토 원칙] 질문에 정확히 답하는 연결재무제표(CFS) 원문만 선택하세요.")
    print("[입력] 1 또는 1,3 (정답 후보 순번), s=질문 보류, q=중단·저장")
    changes = False
    for qid, topic, question, _ in QUESTION_BANK:
        candidates = sorted(grouped[qid], key=lambda r: int(r["candidate_rank"]))
        if not candidates:
            continue
        if any((r.get("manual_verified") or "").strip() in ("0", "1") for r in candidates):
            print(f"[기존 평가 유지] {qid} — 다시 바꾸려면 CSV에서 직접 수정하세요.")
            continue
        print(f"\n=== {qid} / {topic} ===\n질문: {question}")
        for row in candidates:
            print(f"\n({row['candidate_rank']}) [{row['basis_context']}] {row['section']}")
            print(f"원문 ID: {row['chunk_id']}")
            print(f"근거: {row['quote']}")
            print(f"출처: {row['source_url']}")
        while True:
            reply = input("정답 후보 번호 (예: 1,2 / s / q): ").strip().lower()
            if reply in ("s", "q"):
                break
            try:
                picks = {int(piece.strip()) for piece in reply.split(",")}
                allowed = {int(r["candidate_rank"]) for r in candidates}
                if picks and picks <= allowed:
                    break
            except ValueError:
                pass
            print("[입력 오류] 후보 번호만 쉼표로 구분하거나 s/q를 입력하세요.")
        if reply == "q":
            break
        if reply == "s":
            print("[보류] 해당 질문은 검증하지 않은 상태로 남겨둡니다.")
            continue
        note = input("근거 검토 메모 (없으면 Enter): ").strip()
        for r in candidates:
            r["manual_verified"] = "1" if int(r["candidate_rank"]) in picks else "0"
            r["review_notes"] = note if int(r["candidate_rank"]) in picks else ""
        changes = True
        print(f"[평가 반영] {qid} 정답 후보 {', '.join(map(str, sorted(picks)))}")
    if changes:
        import shutil
        backup = review_dir / "gold_review.before_interactive_review.csv"
        if not backup.exists():
            shutil.copy2(path, backup)
        write_csv(path, rows, FIELDS)
        print(f"[평가 저장] {path}")
        print(f"[백업] {backup}")
    else:
        print("[변경 사항 없음]")
    return path


def evaluate(args):
    docs, chunks, digest, snapshot = load_corpus(args.company, args.year, args.basis, args.cache_dir)
    verified = audit_labels(args.review_dir, chunks, digest)
    if not verified:
        raise ValueError("manual_verified=1로 확인된 정답 문단이 없습니다. 먼저 gold_review.csv를 검토하세요.")
    cache_dir = Path(args.cache_dir)
    chunk_lookup = {c["id"]: c for c in chunks}
    model_names = list(dict.fromkeys(args.models))
    print(f"[평가 시작] {len(verified)}개 수동 확인 질문 / {len(chunks)}개 검색 문단 / {len(model_names)}개 임베딩 모델")
    results = []
    top_rows = []
    elapsed_start = time.perf_counter()
    # BM25 is computed ONCE for both models and is independent of embedding size.
    bm25_rankings = {}
    for qid, topic, question, gold in verified:
        bm25_rankings[qid] = [r["id"] for r in retrieve(
            docs, question, basis=args.basis, k=len(chunks),
            structure=True, expansion=True, rerank=True, diversity=False)]

    with httpx.Client(timeout=300.0) as client:
        for model in model_names:
            print(f"[모델] {model}", flush=True)
            vectors, current_digest, reused = base.embed_corpus(
                client, model, chunks, args.batch_size, cache_dir)
            if current_digest != digest:
                raise ValueError("벡터 캐시의 문서 버전이 일치하지 않습니다.")
            if reused:
                print("[기존 벡터 캐시 재사용]", flush=True)
            for i, (qid, topic, question, gold) in enumerate(verified, 1):
                q_vector = base.ollama_embed(client, model, [question])[0]
                dense = base.rank_dense(chunks, vectors, q_vector)
                rankings = {
                    "Dense": dense,
                    "Hybrid_RRF": base.rrf(bm25_rankings[qid], dense),
                }
                if model == model_names[0]:
                    rankings = {"BM25": bm25_rankings[qid], **rankings}
                for method, ids in rankings.items():
                    metrics = base.metrics_at_k(ids, gold, K)
                    method_name = "BM25" if method == "BM25" else f"{method}::{model}"
                    results.append({"question_id": qid, "topic": topic,
                                    "question": question, "method": method_name,
                                    "model": "none" if method == "BM25" else model,
                                    "hit_at_3": metrics["hit_at_3"],
                                    "mrr_at_3": metrics["mrr_at_3"],
                                    "recall_at_3": metrics["recall_at_3"],
                                    "first_gold_rank": metrics["first_gold_rank"],
                                    "gold_count": len(gold)})
                    for rank, cid in enumerate(ids[:K], 1):
                        c = chunk_lookup[cid]
                        top_rows.append({"question_id": qid, "topic": topic,
                                         "method": method_name, "rank": rank,
                                         "chunk_id": cid, "is_gold": int(cid in gold),
                                         "section": c["section"],
                                         "basis_context": c["basis_context"],
                                         "quote_preview": re.sub(r"\s+", " ", c["quote"])[:300],
                                         "source_url": c.get("url", "")})
                print(f"  [{i}/{len(verified)}] {qid} 검색 완료", flush=True)
    summary = []
    for method in dict.fromkeys(r["method"] for r in results):
        records = [r for r in results if r["method"] == method]
        n = len(records)
        summary.append({"method": method, "evaluated_questions": n,
                        "hit_at_3": round(sum(r["hit_at_3"] for r in records)/n, 4),
                        "mrr_at_3": round(sum(r["mrr_at_3"] for r in records)/n, 4),
                        "recall_at_3": round(sum(r["recall_at_3"] for r in records)/n, 4)})
    out = Path(args.review_dir) / ("evaluation_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out/"summary.csv", summary,
              ["method", "evaluated_questions", "hit_at_3", "mrr_at_3", "recall_at_3"])
    write_csv(out/"per_question.csv", results,
              ["question_id", "topic", "question", "method", "model", "hit_at_3",
               "mrr_at_3", "recall_at_3", "first_gold_rank", "gold_count"])
    write_csv(out/"top_results.csv", top_rows,
              ["question_id", "topic", "method", "rank", "chunk_id", "is_gold",
               "section", "basis_context", "quote_preview", "source_url"])
    csv_path = Path(args.review_dir)/"gold_review.csv"
    (out/"metadata.json").write_text(json.dumps({
        "experiment": "exp07c_multitopic_exploratory", "source_snapshot": str(snapshot),
        "corpus_sha256": digest,
        "review_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "company": args.company, "year": args.year, "basis": args.basis,
        "questions_evaluated": len(verified), "topics_evaluated": len({t for _,t,_,_ in verified}),
        "distinct_gold_chunk_ids": len(set().union(*(g for *_,g in verified))),
        "k": K, "models": model_names,
        "bm25": "fea.retrieval.retrieve (same expansion/rerank, unfiltered unknown scope candidates)",
        "dense": "cosine similarity; same corpus and question text for both embedding models",
        "hybrid": "RRF k=60 over BM25 + dense ranked chunk IDs",
        "labeling": "only manual_verified=1 and explicit CFS chunks; unchecked questions excluded",
        "elapsed_seconds": round(time.perf_counter()-elapsed_start, 2),
        "limitations": ["Question set designed using this report, not a held-out evaluation set",
                        "Manual review not independent expert assessment",
                        "Only CFS explicit-context gold spans; unknown context may contain valid answers",
                        "Hit@3/MRR@3 are exact-chunk-ID matches rather than semantic equivalence",
                        "Model size is not the only variable: embeddings differ by training"]
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== Exp07-C: 수동 확인 라벨 기준 검색 비교 ===")
    for r in summary:
        print(f"{r['method']:32s} Hit@3={r['hit_at_3']:.1%} "
              f"MRR@3={r['mrr_at_3']:.3f} Recall@3={r['recall_at_3']:.1%}")
    print(f"[결과 저장] {out}")
    print("[주의] 성능 결과는 확인한 질문과 정답 문단에만 해당하며, 별도/unknown 문맥은 정답으로 자동 인정하지 않습니다.")
    return out, summary


def main():
    p = argparse.ArgumentParser(description="Experiment 07-C: 다주제 OpenDART 검색 비교 / 수동 Gold 검토")
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true", help="질문 16개와 CFS 정답 후보 CSV 작성")
    modes.add_argument("--review", action="store_true", help="질문별 정답 후보를 대화형으로 직접 검토")
    modes.add_argument("--audit", action="store_true", help="수동 검토 라벨 점검 (임베딩 호출 없음)")
    modes.add_argument("--evaluate", action="store_true", help="검토된 질문만 BM25/0.6B/4B/Hybrid 비교")
    p.add_argument("--review-dir", type=Path, help="--prepare 결과 exp07c_dataset_... 경로")
    p.add_argument("--company", default="삼성전자")
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--basis", default="CFS", choices=["CFS"])
    p.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    p.add_argument("--models", nargs="+", default=["qwen3-embedding:0.6b", "qwen3-embedding:4b"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--candidates", type=int, default=5)
    args = p.parse_args()
    if (args.review or args.audit or args.evaluate) and not args.review_dir:
        p.error("--audit/--evaluate에는 --review-dir 경로가 필요합니다.")
    if not 1 <= args.candidates <= 15 or not 1 <= args.batch_size <= 64:
        p.error("--candidates 1~15, --batch-size 1~64만 지원합니다.")
    try:
        if args.prepare:
            prepare(args)
        elif args.review:
            _, chunks, digest, _ = load_corpus(args.company, args.year, args.basis, args.cache_dir)
            audit_labels(args.review_dir, chunks, digest)
            review_interactively(args.review_dir)
        elif args.audit:
            _, chunks, digest, _ = load_corpus(args.company, args.year, args.basis, args.cache_dir)
            audit_labels(args.review_dir, chunks, digest)
        else:
            evaluate(args)
    except (ValueError, FileNotFoundError, httpx.HTTPError, EOFError, KeyboardInterrupt) as e:
        p.exit(1, f"[실험 중단] {e}\n")


if __name__ == "__main__":
    main()
