"""Experiment 07: OpenDART BM25 vs local dense vs RRF hybrid retrieval.

Run from repository root:
  python -m LLM_test.local_ollama_retrieval_exp07 --demo
  python -m LLM_test.local_ollama_retrieval_exp07 --live --company 삼성전자 --year 2024

Dependencies: existing project's httpx and local Ollama qwen3-embedding:0.6b.
This evaluates retrieval only. Weak gold labels are substring matches, not expert annotations.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import time
from datetime import datetime
from getpass import getpass
from pathlib import Path

import httpx

from fea.retrieval import retrieve, structured_chunks

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "experiments" / "ollama_runs" / "exp07_cache"
OUTPUT_ROOT = ROOT / "experiments" / "ollama_runs"
MODEL = "qwen3-embedding:0.6b"
BASIS = "CFS"

# Each anchor is a literal excerpt observed in the 2024 Samsung report.
# Labels are *provisional*: manually review gold_candidates.csv before claims.
SAMSUNG_QUESTIONS = [
    ("Q1", "재고자산의 취득원가와 순실현가능가치는 어떻게 비교하여 평가하나요?", "연결회사는 재고자산을 취득원가와 순실현가능가치 중 낮은 금액으로 측정"),
    ("Q2", "재고자산의 단위원가는 어떤 방법으로 계산하며 예외 품목은 무엇인가요?", "연결회사는 미착품을 제외하고는 재고자산의 단위원가를 평균법"),
    ("Q3", "재고자산의 순실현가능가치를 어떻게 산정하나요?", "추정 판매가격에서 적용 가능한 변동 판매비용을 차감"),
    ("Q4", "재고자산에 적용되는 저가법과 과잉 진부화 평가 방식은 무엇인가요?", "과잉/진부화평가의 두 가지 평가방법"),
    ("Q5", "제품과 재공품의 제조원가에서 제외되는 비용은 무엇인가요?", "유휴 생산설비 원가나 폐기비용은 제외"),
    ("Q6", "2024년 연결 재고자산의 평가 후 장부금액 합계는 얼마인가요?", "51,754,865"),
    ("Q7", "2024년 연결 제품 및 상품의 장부금액은 얼마인가요?", "13,842,276"),
    ("Q8", "전기 연결 재고자산의 평가 후 장부금액 합계는 얼마였나요?", "51,625,874"),
]
DEMO_QUESTIONS = [
    ("D1", "재고자산은 어떤 금액으로 평가하나요?", "취득원가와 순실현가능가치 중 낮은 금액"),
    ("D2", "매출은 언제 인식하나요?", "수행의무가 충족되는 시점"),
    ("D3", "재고자산 원가는 어떤 방법으로 결정하나요?", "재고자산 단위원가는 평균법"),
]


def demo_data():
    text = (
        "연결재무제표 주석\n"
        "1. 재고자산\n"
        "테스트기업은 재고자산을 취득원가와 순실현가능가치 중 낮은 금액으로 평가합니다. "
        "재고자산 단위원가는 평균법을 사용하여 산정합니다.\n"
        "2. 수익인식\n"
        "고객에게 재화의 통제가 이전되어 수행의무가 충족되는 시점에 매출을 인식합니다.\n"
        "3. 기타\n"
        "현금및현금성자산은 요구불예금과 단기예금으로 구성됩니다.\n"
    )
    return [{"id": "DEMO-2024", "company": "00000000", "year": 2024, "basis": "ALL",
             "title": "가상 기업 주석", "text": text, "url": None, "synthetic": True}]


def snapshot_path(cache_dir: Path, company: str, year: int, basis: str) -> Path:
    identity = hashlib.sha256(company.encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"opendart_{year}_{basis}_{identity}.json.gz"


def load_documents(args, cache_dir: Path):
    if args.demo:
        return demo_data(), "가상 예제", DEMO_QUESTIONS
    path = snapshot_path(cache_dir, args.company, args.year, args.basis)
    if path.exists() and not args.refresh:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            saved = json.load(f)
        print(f"[OpenDART 스냅샷 재사용] {path}", flush=True)
        return saved["documents"], saved["company"], SAMSUNG_QUESTIONS
    from fea.live import load_public
    key = getpass("OpenDART API 키 (표시되지 않음): ").strip()
    if not key:
        raise ValueError("OpenDART API 키가 비어 있습니다.")
    data = load_public(key, args.company, args.year, args.basis)
    docs = [d for d in data["documents"] if d["year"] == args.year]
    if not docs:
        raise ValueError("선택 연도 사업보고서를 수집하지 못했습니다.")
    company = data["companies"][0]["name"]
    path.parent.mkdir(parents=True, exist_ok=True)
    # Only the extracted *public* text is stored; never the OpenDART key.
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"company": company, "year": args.year, "basis": args.basis,
                   "documents": docs}, f, ensure_ascii=False)
    print(f"[OpenDART 공개 공시 스냅샷 저장] {path}", flush=True)
    return docs, company, SAMSUNG_QUESTIONS


def relevant_corpus(documents, basis):
    chunks = []
    seen = set()
    for doc in documents:
        for chunk in structured_chunks(doc):
            if chunk["basis_context"] in ("CFS", "OFS") and chunk["basis_context"] != basis:
                continue
            if chunk["id"] in seen:
                raise ValueError("중복된 청크 ID입니다.")
            seen.add(chunk["id"])
            chunks.append(chunk)
    return chunks


def corpus_digest(chunks):
    h = hashlib.sha256()
    for c in chunks:
        h.update(c["id"].encode("utf-8"))
        h.update(b"\x00")
        h.update(c["quote"].encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def normalize(v):
    norm = math.sqrt(sum(float(x) * float(x) for x in v))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("임베딩 벡터의 값이 유효하지 않습니다.")
    return [float(x) / norm for x in v]


def ollama_embed(client: httpx.Client, model: str, inputs: list[str]):
    r = client.post("http://127.0.0.1:11434/api/embed", json={
        "model": model, "input": inputs, "truncate": False
    })
    r.raise_for_status()
    payload = r.json()
    vectors = payload.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(inputs):
        raise ValueError("Ollama의 임베딩 개수가 입력 개수와 일치하지 않습니다.")
    out = [normalize(vec) for vec in vectors]
    if len({len(vec) for vec in out}) != 1:
        raise ValueError("임베딩 차원이 일치하지 않습니다.")
    return out


def embed_corpus(client, model, chunks, batch_size, cache_dir):
    digest = corpus_digest(chunks)
    model_tag = hashlib.sha256(model.encode("utf-8")).hexdigest()[:12]
    path = cache_dir / f"vectors_{model_tag}_{digest[:18]}.json.gz"
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as f:
            saved = json.load(f)
        vecs = saved.get("vectors", [])
        if (saved.get("model") == model and saved.get("corpus_sha256") == digest
                and len(vecs) == len(chunks) and vecs
                and len({len(v) for v in vecs}) == 1):
            print(f"[벡터 캐시 재사용] {path}", flush=True)
            return [normalize(v) for v in vecs], digest, True
        raise ValueError("임베딩 캐시 검증 실패. 파일을 확인하거나 제거해 주세요.")
    vectors = []
    total_batches = math.ceil(len(chunks) / batch_size)
    print(f"[임베딩 생성] 문단 {len(chunks)}개 / 배치 {total_batches}개", flush=True)
    for offset in range(0, len(chunks), batch_size):
        batch = chunks[offset:offset+batch_size]
        # Use exactly the same text for BM25 and dense search (no instruction prefix).
        vectors.extend(ollama_embed(client, model, [c["quote"] for c in batch]))
        current = offset // batch_size + 1
        if current == 1 or current == total_batches or current % 10 == 0:
            print(f"  [임베딩 진행] {current}/{total_batches} 배치", flush=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"model": model, "corpus_sha256": digest, "vectors": vectors}, f)
    print(f"[벡터 캐시 저장] {path}", flush=True)
    return vectors, digest, False


def rank_dense(chunks, vectors, q_vector):
    if len(chunks) != len(vectors):
        raise ValueError("청크/벡터 개수 불일치")
    ranked = sorted(zip(chunks, vectors),
                    key=lambda cv: (-sum(a*b for a, b in zip(cv[1], q_vector)), cv[0]["id"]))
    return [chunk["id"] for chunk, _ in ranked]


def rrf(bm25_ids, dense_ids, constant=60):
    scores = {}
    for ranking in (bm25_ids, dense_ids):
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (constant + rank)
    return sorted(scores, key=lambda cid: (-scores[cid], cid))


def metrics_at_k(ranked_ids, gold_ids, k=3):
    gold = set(gold_ids)
    top = ranked_ids[:k]
    matches = [i + 1 for i, cid in enumerate(top) if cid in gold]
    return {"hit_at_3": int(bool(matches)),
            "mrr_at_3": 1.0 / matches[0] if matches else 0.0,
            "recall_at_3": len(matches) / len(gold) if gold else 0.0,
            "first_gold_rank": matches[0] if matches else ""}


def write_csv(path: Path, rows, columns):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(args, client=None):
    cache_dir = Path(args.cache_dir)
    docs, company, questions = load_documents(args, cache_dir)
    if not args.demo and company != "삼성전자":
        raise ValueError("기본 평가 질문과 정답 문구는 삼성전자 2024년 사업보고서 전용입니다. 다른 기업에는 별도 평가셋이 필요합니다.")
    if not args.demo and args.year != 2024:
        raise ValueError("기본 평가 질문은 삼성전자 2024년 보고서 전용입니다.")
    chunks = relevant_corpus(docs, args.basis)
    if not chunks:
        raise ValueError("검색 가능한 문단이 없습니다.")
    print(f"[데이터] {company} / {args.year} / {args.basis} / 검색 문단 {len(chunks)}개")
    print(f"[모델] {args.model} (로컬 Ollama, 답변 생성 LLM은 사용하지 않음)")

    owned_client = client is None
    client = client or httpx.Client(timeout=300.0)
    try:
        started = time.perf_counter()
        vectors, digest, reused = embed_corpus(client, args.model, chunks, args.batch_size, cache_dir)
        chunk_index = {c["id"]: c for c in chunks}
        ranking_rows, score_rows, gold_rows = [], [], []
        skipped = []
        for qid, question, anchor in questions:
            gold = {c["id"] for c in chunks if anchor in c["quote"] and (args.demo or c["basis_context"] == args.basis)}
            if not gold:
                skipped.append(qid)
                print(f"[평가 제외] {qid}: 연결 주석 내 정답 앵커를 찾지 못했습니다. 실제 문서/라벨을 확인하세요.")
                continue
            for cid in sorted(gold):
                c = chunk_index[cid]
                gold_rows.append({"question_id": qid, "question": question, "anchor": anchor,
                                  "gold_chunk_id": cid, "section": c["section"],
                                  "context": c["basis_context"], "quote": c["quote"],
                                  "manual_verified": ""})
            # BM25 is the existing fea.retrieval pipeline including its expansion & re-ranking.
            bm25 = retrieve(docs, question, basis=args.basis, k=len(chunks),
                            structure=True, expansion=True, rerank=True, diversity=False)
            bm25_ids = [c["id"] for c in bm25]
            q_vec = ollama_embed(client, args.model, [question])[0]
            dense_ids = rank_dense(chunks, vectors, q_vec)
            hybrid_ids = rrf(bm25_ids, dense_ids)
            print(f"\n[{qid}] {question}")
            for method, ids in (("BM25", bm25_ids), ("Dense", dense_ids), ("Hybrid_RRF", hybrid_ids)):
                m = metrics_at_k(ids, gold, args.k)
                score_rows.append({"question_id": qid, "method": method, **m,
                                   "gold_count": len(gold)})
                print(f"  {method:10s} Hit@{args.k}={m['hit_at_3']} MRR@{args.k}={m['mrr_at_3']:.3f} "
                      f"Recall@{args.k}={m['recall_at_3']:.3f}")
                for rank, cid in enumerate(ids[:args.k], 1):
                    c = chunk_index[cid]
                    ranking_rows.append({"question_id": qid, "method": method, "rank": rank,
                                         "chunk_id": cid, "is_gold": int(cid in gold),
                                         "section": c["section"], "basis_context": c["basis_context"],
                                         "quote_preview": re.sub(r"\s+", " ", c["quote"])[:300],
                                         "source_url": c.get("url", "")})
        if not score_rows:
            raise ValueError("평가 가능한 정답 문단을 찾지 못했습니다. 테스트가 성공했다고 보고하지 않습니다.")
        output = OUTPUT_ROOT / ("exp07_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        output.mkdir(parents=True, exist_ok=False)
        columns = ["question_id", "method", "hit_at_3", "mrr_at_3", "recall_at_3", "first_gold_rank", "gold_count"]
        write_csv(output/"per_question.csv", score_rows, columns)
        write_csv(output/"top_results.csv", ranking_rows,
                  ["question_id", "method", "rank", "chunk_id", "is_gold", "section", "basis_context", "quote_preview", "source_url"])
        write_csv(output/"gold_candidates.csv", gold_rows,
                  ["question_id", "question", "anchor", "gold_chunk_id", "section", "context", "quote", "manual_verified"])
        summary = []
        for method in ("BM25", "Dense", "Hybrid_RRF"):
            records = [r for r in score_rows if r["method"] == method]
            n = len(records)
            summary.append({"method": method, "evaluated_questions": n,
                            "hit_at_3": round(sum(r["hit_at_3"] for r in records)/n, 4),
                            "mrr_at_3": round(sum(r["mrr_at_3"] for r in records)/n, 4),
                            "recall_at_3": round(sum(r["recall_at_3"] for r in records)/n, 4)})
        write_csv(output/"summary.csv", summary,
                  ["method", "evaluated_questions", "hit_at_3", "mrr_at_3", "recall_at_3"])
        (output/"metadata.json").write_text(json.dumps({
            "experiment": "exp07_retrieval_pilot", "model": args.model,
            "company": company, "year": args.year, "basis": args.basis,
            "corpus_chunks": len(chunks), "corpus_sha256": digest,
            "embeddings_cache_reused": reused, "batch_size": args.batch_size,
            "k": args.k, "skipped_question_ids": skipped,
            "labeling": "provisional phrase-match weak labels, CFS context only; manual verification required",
            "bm25": "fea.retrieval.retrieve with expansion and re-rank, no diversity",
            "hybrid": "RRF of BM25 and cosine dense rankings (k=60)",
            "elapsed_seconds": round(time.perf_counter()-started, 2)
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n=== Experiment 07 요약 (잠정 정답 문구 기준) ===")
        for s in summary:
            print(f"{s['method']:10s} | Hit@{args.k} {s['hit_at_3']:.1%} | "
                  f"MRR@{args.k} {s['mrr_at_3']:.3f} | Recall@{args.k} {s['recall_at_3']:.1%}")
        print(f"[결과 저장] {output}")
        print("[중요] gold_candidates.csv의 정답 문단을 수동 확인해야 합니다. "
              "이번 파일럿의 정답 문구/질문은 재고자산 주석 중심이므로 일반화 불가.")
        return output, summary
    finally:
        if owned_client:
            client.close()


def main():
    parser = argparse.ArgumentParser(description="Exp07: BM25 / Dense / Hybrid retrieval on the same OpenDART chunks")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="가상 문서로 로컬 임베딩 호출 검증")
    mode.add_argument("--live", action="store_true", help="실제 OpenDART 삼성전자 2024 공시")
    parser.add_argument("--company", default="삼성전자")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--basis", choices=["CFS"], default=BASIS)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--k", type=int, default=3, choices=[3], help="현재 평가는 @3 고정")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--refresh", action="store_true", help="저장한 OpenDART 공시 스냅샷을 새로 수집")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 64:
        parser.error("--batch-size는 1~64만 지원합니다.")
    try:
        run(args)
    except (httpx.HTTPError, ValueError) as exc:
        parser.exit(1, f"[실험 중단] {exc}\n[확인] Ollama 서버 실행 및 'ollama pull {args.model}' 여부를 확인하세요.\n")


if __name__ == "__main__":
    main()
