"""Experiment 07-E — audit scope provenance in a cached OpenDART report.

Put at LLM_test/exp07e_scope_provenance.py.

  python -m LLM_test.exp07e_scope_provenance --prepare --review-dir ...
  python -m LLM_test.exp07e_scope_provenance --review --audit-dir ... --focus R14 R16
  python -m LLM_test.exp07e_scope_provenance --show --audit-dir ... --segment-id <id>
  python -m LLM_test.exp07e_scope_provenance --audit --audit-dir ...
  python -m LLM_test.exp07e_scope_provenance --evaluate --audit-dir ... --model qwen3-embedding:4b

All offsets are into the *saved extracted plain text*, NOT into the DART HTML/XML.
Suggested scope from an anchor is NOT evidence of scope and is NEVER trusted for
filtering. Only manually reviewed segments can add/reject scope; old explicit-CFS
metadata is retained as an imperfect, pre-existing baseline. Original source,
question bank, labels and corpus/vector cache are never modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import httpx
from LLM_test import exp07c_multitopic_evaluation as c
from LLM_test import exp07a_retrieval_benchmark as base
from fea.retrieval import retrieve

ANCHOR_COLUMNS = ["source_id", "offset", "line", "kind", "suggested_scope", "before", "after"]
SEG_COLUMNS = ["segment_id", "source_id", "start", "end", "anchor_kind", "anchor_line",
               "suggested_scope", "manual_scope", "review_notes", "chunk_count",
               "previous_heading", "sample_start", "sample_end", "document_sha256"]
SCOPE_CHOICES = ("CFS", "OFS", "unknown", "")

# Titles deliberately require a COMPLETE physical line. A sentence containing
# e.g. '연결재무제표' is not a statement-boundary marker.
FILE_LINE = re.compile(r"^\[원문 파일:\s*.+\]$")
NOTE_TITLE = re.compile(r"^(?:(?:\d+|[IVXⅠⅡⅢⅣⅤ]+)[.．)]\s*)?(연결\s*)?재무제표\s*주석$")
STATEMENT_TITLE = re.compile(r"^(?:(?:\d+|[IVXⅠⅡⅢⅣⅤ]+)[.．)]\s*)?(연결\s*)?재무제표$")
STATEMENT_HEAD = re.compile(r"^(연결\s*)?(재무상태표|손익계산서|포괄손익계산서|자본변동표|현금흐름표)$")
AUDIT_REPORT = re.compile(r"^(독립된\s*)?(외부)?감사인의?\s*(?:감사|검토)보고서$")


def norm_line(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def classify_heading(line: str):
    if FILE_LINE.fullmatch(line):
        return "file_boundary", "unknown"
    m = NOTE_TITLE.fullmatch(line)
    if m:
        return "notes_title", "CFS" if m.group(1) else "OFS"
    m = STATEMENT_TITLE.fullmatch(line)
    if m:
        return "statement_title", "CFS" if m.group(1) else "OFS"
    m = STATEMENT_HEAD.fullmatch(line)
    if m:
        # A bare '재무상태표' does not prove OFS; it can occur in a group report.
        return "statement_table_title", "CFS" if m.group(1) else "unknown"
    if AUDIT_REPORT.fullmatch(line):
        return "audit_report_title", "unknown"
    return None


def markers_for(doc):
    """Returns evidence candidates, not asserted statement scopes."""
    text = doc["text"]
    markers = []
    for match in re.finditer(r"(?m)^([^\n]*)", text):
        original_line = match.group(1)
        line = norm_line(original_line)
        if not line or len(line) > 110:
            continue
        found = classify_heading(line)
        if not found:
            continue
        kind, guess = found
        off = match.start()
        markers.append({"source_id": doc["id"], "offset": off,
                        "line": line, "kind": kind, "suggested_scope": guess,
                        "before": norm_line(text[max(0, off - 240):off])[-240:],
                        "after": norm_line(text[match.end():match.end() + 240])[:240]})
    # Dedup protects against weird regex zero-length line matches.
    return sorted({m["offset"]: m for m in markers}.values(), key=lambda x: x["offset"])


def overlaps(start, end, lo, hi):
    return max(start, lo) < min(end, hi)


def make_segments(doc, chunks, markers):
    """Coverage partition. Candidate headings start segments, but DO NOT label them."""
    text = doc["text"]
    positions = [0] + [int(m["offset"]) for m in markers if int(m["offset"]) > 0] + [len(text)]
    positions = sorted(set(positions))
    by_pos = {int(m["offset"]): m for m in markers}
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    out = []
    previous = "[문서 시작]"
    relevant = [ch for ch in chunks if ch["source_id"] == doc["id"]]
    for left, right in zip(positions, positions[1:]):
        if right <= left:
            continue
        marker = by_pos.get(left)
        heading = marker["line"] if marker else "[문서 시작 / 제목 미확인]"
        kind = marker["kind"] if marker else "document_start"
        guess = marker["suggested_scope"] if marker else "unknown"
        # Protect against accidentally treating unverified recommendations as labels.
        payload = f'{doc["id"]}|{left}|{right}|{sha}'
        segment_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        piece = text[left:right]
        out.append({"segment_id": segment_id, "source_id": doc["id"],
                    "start": left, "end": right, "anchor_kind": kind,
                    "anchor_line": heading, "suggested_scope": guess,
                    "manual_scope": "", "review_notes": "",
                    "chunk_count": sum(overlaps(ch["start"], ch["end"], left, right) for ch in relevant),
                    "previous_heading": previous,
                    "sample_start": norm_line(piece[:450]),
                    "sample_end": norm_line(piece[-350:]),
                    "document_sha256": sha})
        previous = heading
    return out


def read_csv(p):
    with Path(p).open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(p, rows, fields):
    with Path(p).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def doc_digest(docs):
    h = hashlib.sha256()
    for d in docs:
        h.update(d["id"].encode("utf-8"))
        h.update(b"\0")
        h.update(hashlib.sha256(d["text"].encode("utf-8")).digest())
    return h.hexdigest()


def load_source(audit_dir, args):
    meta_path = Path(audit_dir) / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"준비된 audit 디렉터리가 없습니다: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    docs, chunks, digest, snapshot = c.load_corpus(meta["company"], meta["year"], meta["basis"], args.cache_dir)
    if digest != meta["corpus_sha256"] or doc_digest(docs) != meta["document_sha256"]:
        raise ValueError("원문/청크 스냅샷이 변경되었습니다. 기존 범위 검토와 비교할 수 없습니다.")
    if hashlib.sha256((Path(meta["review_dir"]) / "gold_review.csv").read_bytes()).hexdigest() != meta["gold_csv_sha256"]:
        raise ValueError("Gold 라벨 파일이 변경되었습니다. 재검토/재실행 후 최신 스냅샷으로 준비해 주세요.")
    segments = read_csv(Path(audit_dir) / "scope_segments.csv")
    by_doc = {d["id"]: d for d in docs}
    seg_ids = set()
    for s in segments:
        sid = s["segment_id"]
        if sid in seg_ids:
            raise ValueError("중복 segment ID")
        seg_ids.add(sid)
        if s["manual_scope"] not in SCOPE_CHOICES:
            raise ValueError(f"잘못된 manual_scope: {s['manual_scope']}")
        d = by_doc.get(s["source_id"])
        if not d or s["document_sha256"] != hashlib.sha256(d["text"].encode("utf-8")).hexdigest():
            raise ValueError("scope_segments.csv가 원문과 불일치합니다.")
        if not 0 <= int(s["start"]) < int(s["end"]) <= len(d["text"]):
            raise ValueError("범위 오프셋이 잘못되었습니다.")
        fingerprint = hashlib.sha256(f'{d["id"]}|{s["start"]}|{s["end"]}|{s["document_sha256"]}'.encode("utf-8")).hexdigest()[:16]
        if sid != fingerprint:
            raise ValueError("scope_segments.csv의 ID/범위가 변경되었습니다.")
    return meta, docs, chunks, segments


def focus_chunk_ids(meta, questions):
    """Existing D-run top results + verified Gold; never treated as newly verified labels."""
    focus = set()
    review_dir = Path(meta["review_dir"])
    for row in read_csv(review_dir / "gold_review.csv"):
        if row.get("question_id") in questions and row.get("manual_verified", "").strip() == "1":
            focus.add(row["chunk_id"])
    d_dirs = sorted(review_dir.glob("evaluation_scope_*"))
    if d_dirs:
        path = d_dirs[-1] / "top_results.csv"
        if path.exists():
            for row in read_csv(path):
                if row.get("question_id") in questions:
                    focus.add(row["chunk_id"])
    return focus


def prioritize_segments(meta, chunks, segments, questions):
    if not questions:
        return list(segments)
    focus_ids = focus_chunk_ids(meta, questions)
    targeted = [ch for ch in chunks if ch["id"] in focus_ids]
    chosen = []
    for s in segments:
        if any(ch["source_id"] == s["source_id"] and overlaps(ch["start"], ch["end"], int(s["start"]), int(s["end"])) for ch in targeted):
            chosen.append(s)
    return chosen


def prepare(args):
    docs, chunks, digest, snapshot = c.load_corpus(args.company, args.year, args.basis, args.cache_dir)
    verified = c.audit_labels(args.review_dir, chunks, digest)
    if not verified:
        raise ValueError("검증된 Gold 라벨이 없습니다.")
    markers = []
    segments = []
    for doc in docs:
        ms = markers_for(doc)
        markers.extend(ms)
        segments.extend(make_segments(doc, chunks, ms))
    dest = Path(args.review_dir) / ("scope_audit_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    dest.mkdir(parents=True, exist_ok=False)
    write_csv(dest / "scope_anchors.csv", markers, ANCHOR_COLUMNS)
    write_csv(dest / "scope_segments.csv", segments, SEG_COLUMNS)
    meta = {"experiment": "Exp07-E raw_text_scope_provenance", "review_dir": str(Path(args.review_dir).resolve()),
            "company": args.company, "year": args.year, "basis": args.basis, "snapshot": str(snapshot),
            "corpus_sha256": digest, "document_sha256": doc_digest(docs),
            "gold_csv_sha256": hashlib.sha256((Path(args.review_dir) / "gold_review.csv").read_bytes()).hexdigest(),
            "chunks": len(chunks), "anchors": len(markers), "segments": len(segments),
            "limitations": [
                "Offsets refer only to extracted text, not HTML/XML node locations",
                "Suggested scope is untrusted and never used as a verified label",
                "Section titles can appear in tables of contents or repeat in a single document",
                "Only a person can mark segment CFS/OFS/unknown after checking the neighboring source",
                "Chunk crossing an audited segment boundary remains unresolved",
                "Metadata-only CFS is an imperfect pre-existing baseline, not independently verified",
                "No financial figures, source files, or Gold labels are edited",
            ]}
    (dest / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    old = Counter(ch.get("basis_context", "unknown") for ch in chunks)
    print(f"[원문 검토 준비] 전체 {len(chunks)}개 문단 | 기존 CFS={old['CFS']}, unknown={old['unknown']}")
    print(f"[제목 단서] {len(markers)}개 | 검토 구간 {len(segments)}개")
    for m in markers[:35]:
        print(f"  {m['offset']:>8} {m['kind']:<22} {m['suggested_scope']:<7} {m['line'][:75]}")
    if len(markers) > 35:
        print(f"  ... 추가 {len(markers)-35}개 단서는 scope_anchors.csv 참조")
    print(f"[결과 저장] {dest}")
    print("[다음] --review --audit-dir 위경로 --focus R14 R16")
    print("[경고] 제목 단서만으로 연결/별도 범위를 확정하지 않았습니다.")
    return dest


def review(args):
    audit_dir = Path(args.audit_dir)
    meta, docs, chunks, segments = load_source(audit_dir, args)
    by_doc = {d["id"]: d for d in docs}
    targets = prioritize_segments(meta, chunks, segments, set(args.focus))
    print(f"[구간 수동 검토] 전체 {len(segments)}개 중 대상 {len(targets)}개")
    print("[입력] c=CFS 확인, o=OFS 확인, u=미확인(검토했지만 결정불가), s=건너뛰기, q=중단")
    print("[원칙] 자동 단서는 정답 아님. 이 구간 전체가 같은 연결/별도 범위인지 확인할 때만 c/o.")
    changed = False
    for s in targets:
        if s["manual_scope"]:
            continue
        d = by_doc[s["source_id"]]
        start, end = int(s["start"]), int(s["end"])
        print("\n" + "=" * 76)
        print(f"구간 {s['segment_id']} / {start:,}~{end:,} / 길이 {end-start:,}자 / 포함·겹침 청크 {s['chunk_count']}개")
        print(f"원문 파일: {d['id']} / {d.get('url', '')}")
        print(f"직전 제목: {s['previous_heading']}")
        print(f"현재 제목: {s['anchor_line']}  (단서: {s['suggested_scope']}, {s['anchor_kind']})")
        print(f"첫 부분: {s['sample_start'][:450]}")
        print(f"끝 부분: {s['sample_end'][-350:]}")
        print("주의: 단서와 발췌만으로 구간 전체 범위가 불확실하면 반드시 u 또는 s를 선택하세요.")
        while True:
            choice = input("범위 확인 [c/o/u/s/q]: ").strip().lower()
            if choice in ("c", "o", "u", "s", "q"):
                break
            print("c/o/u/s/q 중 하나를 입력하세요.")
        if choice == "q":
            break
        if choice == "s":
            continue
        s["manual_scope"] = {"c":"CFS", "o":"OFS", "u":"unknown"}[choice]
        s["review_notes"] = input("원문 확인 근거 메모 (권장): ").strip()
        changed = True
    if changed:
        write_csv(audit_dir / "scope_segments.csv", segments, SEG_COLUMNS)
        print(f"[검토 저장] {audit_dir / 'scope_segments.csv'}")
    else:
        print("[변경 없음]")
    print("[다음] --audit --audit-dir 위경로")


def segment_index(segments):
    by_doc = defaultdict(list)
    for s in segments:
        by_doc[s["source_id"]].append(s)
    for d in by_doc:
        by_doc[d].sort(key=lambda r: int(r["start"]))
        for prev, now in zip(by_doc[d], by_doc[d][1:]):
            if int(prev["end"]) != int(now["start"]):
                raise ValueError(f"검토 구간이 연결되지 않습니다: {d}")
    return by_doc


def audited_scope(ch, by_doc):
    """Returns manually verified scope iff a chunk is fully within ONE reviewed segment."""
    covered = [s for s in by_doc.get(ch["source_id"], [])
               if int(s["start"]) <= ch["start"] and ch["end"] <= int(s["end"])]
    if len(covered) != 1:
        return "unresolved_boundary"
    return covered[0]["manual_scope"] or "not_reviewed"


def audit(args):
    meta, docs, chunks, segments = load_source(args.audit_dir, args)
    idx = segment_index(segments)
    rows = []
    for ch in chunks:
        scope = audited_scope(ch, idx)
        rows.append({"chunk_id": ch["id"], "original_basis": ch.get("basis_context", "unknown"),
                     "manual_basis": scope, "section": ch.get("section", ""),
                     "start": ch["start"], "end": ch["end"],
                     "quote_preview": norm_line(ch["quote"])[:180]})
    count = Counter(r["manual_basis"] for r in rows)
    initially_unknown = Counter(r["manual_basis"] for r in rows if r["original_basis"] == "unknown")
    old_explicit = {ch["id"] for ch in chunks if ch.get("basis_context") == "CFS"}
    confirmed_additions = {r["chunk_id"] for r in rows if r["manual_basis"] == "CFS" and r["original_basis"] == "unknown"}
    manual_ofs = {r["chunk_id"] for r in rows if r["manual_basis"] == "OFS"}
    conflicts = sorted(old_explicit & manual_ofs)
    output = Path(args.audit_dir) / "chunk_scope_audit.csv"
    write_csv(output, rows, ["chunk_id", "original_basis", "manual_basis", "section", "start", "end", "quote_preview"])
    print(f"[구간 검토] 총 {len(segments)}개 / CFS {sum(s['manual_scope']=='CFS' for s in segments)}개 / OFS {sum(s['manual_scope']=='OFS' for s in segments)}개 / 미확인 {sum(s['manual_scope']=='unknown' for s in segments)}개")
    print(f"[문단 검토] {dict(count)}")
    print(f"[기존 unknown에 대한 검토] {dict(initially_unknown)}")
    print(f"[기존 CFS + 새로 확인된 unknown CFS] {len(old_explicit)} + {len(confirmed_additions)}개 (중복 없음)")
    print(f"[충돌] 기존 CFS로 기록됐으나 수동 OFS 판정: {len(conflicts)}개")
    print(f"[결과 저장] {output}")
    if conflicts:
        print("[주의] 메타데이터 충돌! 추론 대신 실제 원문·섹션 경계를 다시 확인하세요.")
    return rows, confirmed_additions, manual_ofs, conflicts


def show_segment(args):
    meta, docs, chunks, segments = load_source(args.audit_dir, args)
    selected = [s for s in segments if s["segment_id"] == args.segment_id]
    if not selected:
        raise ValueError("존재하지 않는 segment_id입니다. --audit-dir의 scope_segments.csv를 확인하세요.")
    seg = selected[0]
    doc = next(d for d in docs if d["id"] == seg["source_id"])
    lo, hi = int(seg["start"]), int(seg["end"])
    text = doc["text"]
    width = args.window
    if width < 100 or width > 20000:
        raise ValueError("--window는 100~20000자로 지정하세요.")
    print(f"[범위] {doc['id']} / {lo:,}~{hi:,} ({hi-lo:,}자)")
    print(f"[시작 제목] {seg['anchor_line']} | 자동 단서={seg['suggested_scope']} | 사람 확인={seg['manual_scope'] or '미확인'}")
    if hi - lo <= 2 * width:
        print(text[lo:hi])
    else:
        print(f"\n[시작 {width}자]\n{text[lo:lo+width]}")
        print(f"\n[종료 {width}자]\n{text[hi-width:hi]}")
        print(f"\n[중앙에서 {width}자]\n{text[max(lo,(lo+hi)//2-width//2):min(hi,(lo+hi)//2+width//2)]}")
    print("[주의] 일부 발췌만으로 긴 구간 전체의 연결/별도 범위가 확정되지 않을 수 있습니다.")


def evaluate(args):
    meta, docs, chunks, segments = load_source(args.audit_dir, args)
    verified = c.audit_labels(meta["review_dir"], chunks, meta["corpus_sha256"])
    if not verified:
        raise ValueError("평가 가능한 Gold가 없습니다.")
    _, add_cfs, confirmed_ofs, conflicts = audit(args)
    if conflicts:
        raise ValueError("기존 CFS 메타데이터와 수동 OFS 판단이 충돌하므로 평가를 중단합니다.")
    if not add_cfs and not confirmed_ofs:
        raise ValueError("수동 검토 결과가 없습니다. --review로 구간을 확인한 뒤 평가하세요.")
    old_ids = {ch["id"] for ch in chunks if ch.get("basis_context") == "CFS"}
    revised_ids = (old_ids | add_cfs) - confirmed_ofs
    lookup = {ch["id"]: ch for ch in chunks}
    results, top = [], []
    with httpx.Client(timeout=300.0) as client:
        model = args.model
        model_tag = hashlib.sha256(model.encode("utf-8")).hexdigest()[:12]
        expected_cache = Path(args.cache_dir) / f"vectors_{model_tag}_{meta['corpus_sha256'][:18]}.json.gz"
        if not expected_cache.exists():
            raise FileNotFoundError(f"원문과 모델이 같은 벡터 캐시가 없습니다: {expected_cache}")
        vectors, digest, reused = base.embed_corpus(client, model, chunks, args.batch_size, args.cache_dir)
        if digest != meta["corpus_sha256"] or not reused:
            raise ValueError("동일 임베딩 캐시가 없어 비교를 중단합니다. 기존 원문/모델 설정을 확인하세요.")
        print(f"[벡터 캐시 검증] {model} / 재사용 성공")
        for i, (qid, topic, question, gold) in enumerate(verified, 1):
            bm25 = [r["id"] for r in retrieve(docs, question, basis="CFS", k=len(chunks),
                         structure=True, expansion=True, rerank=True, diversity=False)]
            q_vec = base.ollama_embed(client, model, [question])[0]
            dense = base.rank_dense(chunks, vectors, q_vec)
            for name, allowed in [("original_hybrid", None), ("old_metadata_CFS_only", old_ids),
                                  ("CFS_plus_reviewed", revised_ids)]:
                if allowed is None:
                    ranks = base.rrf(bm25, dense)
                else:
                    ranks = base.rrf([x for x in bm25 if x in allowed], [x for x in dense if x in allowed])
                metrics = base.metrics_at_k(ranks, gold, 3)
                results.append({"question_id": qid, "topic": topic, "method": name,
                                **metrics, "gold_count": len(gold)})
                for rank, cid in enumerate(ranks[:3], 1):
                    ch = lookup[cid]
                    top.append({"question_id": qid, "method": name, "rank": rank,
                                "chunk_id": cid, "is_gold": int(cid in gold),
                                "original_basis": ch.get("basis_context", "unknown"),
                                "quote_preview": norm_line(ch["quote"])[:220]})
            print(f"  [{i}/{len(verified)}] {qid}", flush=True)
    by_method = defaultdict(list)
    for r in results:
        by_method[r["method"]].append(r)
    summarized = []
    for name, v in by_method.items():
        summarized.append({"method": name, "questions": len(v),
                           "hit_at_3": round(sum(x["hit_at_3"] for x in v)/len(v), 4),
                           "mrr_at_3": round(sum(x["mrr_at_3"] for x in v)/len(v), 4),
                           "recall_at_3": round(sum(x["recall_at_3"] for x in v)/len(v), 4)})
    dest = Path(args.audit_dir) / ("evaluation_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    dest.mkdir(parents=True, exist_ok=False)
    write_csv(dest / "summary.csv", summarized, ["method", "questions", "hit_at_3", "mrr_at_3", "recall_at_3"])
    write_csv(dest / "per_question.csv", results, ["question_id", "topic", "method", "hit_at_3", "mrr_at_3", "recall_at_3", "first_gold_rank", "gold_count"])
    write_csv(dest / "top_results.csv", top, ["question_id", "method", "rank", "chunk_id", "is_gold", "original_basis", "quote_preview"])
    (dest / "metadata.json").write_text(json.dumps({"model": args.model, "scope_audit_dir": str(args.audit_dir),
               "original_CFS_count": len(old_ids), "manual_added_CFS_count": len(add_cfs),
               "revised_CFS_count": len(revised_ids), "evaluated_questions": len(verified),
               "note": "CFS_plus_reviewed is a partial review, not a fully verified corpus. Exact-ID Gold may penalize semantically identical passages."}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== Experiment 07-E / scope provenance (partial review) ===")
    for r in summarized:
        print(f"{r['method']:<26} Hit@3={r['hit_at_3']:.1%} MRR@3={r['mrr_at_3']:.3f} Recall@3={r['recall_at_3']:.1%}")
    print("[주의] 기존 CFS 메타데이터와 일부 수동 검토 결과를 결합한 탐색적 비교입니다.")
    print(f"[결과 저장] {dest}")
    return dest


def main():
    p = argparse.ArgumentParser(description="Experiment 07-E: DART statement-scope provenance audit")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--review", action="store_true")
    mode.add_argument("--audit", action="store_true")
    mode.add_argument("--show", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    p.add_argument("--review-dir", type=Path, default=None, help="Exp07-C dataset directory (for --prepare)")
    p.add_argument("--audit-dir", type=Path, default=None, help="Exp07-E scope_audit_* directory")
    p.add_argument("--focus", nargs="*", default=["R14", "R16"], help="Question IDs for focused segment review")
    p.add_argument("--company", default="삼성전자")
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--basis", choices=["CFS"], default="CFS")
    p.add_argument("--cache-dir", type=Path, default=base.DEFAULT_CACHE)
    p.add_argument("--segment-id", default=None, help="Scope segment ID to inspect with --show")
    p.add_argument("--window", default=1200, type=int, help="Chars per inspection excerpt, 100-20000")
    p.add_argument("--model", default="qwen3-embedding:4b")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    if args.prepare and not args.review_dir:
        p.error("--prepare에는 --review-dir이 필요합니다.")
    if not args.prepare and not args.audit_dir:
        p.error("--review/--audit/--show/--evaluate에는 --audit-dir이 필요합니다.")
    if args.show and not args.segment_id:
        p.error("--show에는 --segment-id가 필요합니다.")
    try:
        if args.prepare:
            prepare(args)
        elif args.review:
            review(args)
        elif args.audit:
            audit(args)
        elif args.show:
            show_segment(args)
        else:
            evaluate(args)
    except (ValueError, FileNotFoundError, httpx.HTTPError, KeyboardInterrupt) as exc:
        p.exit(1, f"[실험 중단] {exc}\n")


if __name__ == "__main__":
    main()
