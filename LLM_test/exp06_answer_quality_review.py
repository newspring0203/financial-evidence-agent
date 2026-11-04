"""Interactive, manual answer-quality review for Experiment 06.

Place in financial-evidence-agent/LLM_test/ and run from repository root:
  python -m LLM_test.exp06_answer_quality_review --dir experiments/ollama_runs/exp06_20261104_082828_185645

Requires only Python standard library. It updates an existing renamed
answer_quality_review.csv, preserving other fields and making a one-time backup.
Review scores are entered by the reviewer, never guessed automatically.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

CRITERIA = (
    "평균법으로 단위원가 결정",
    "미착품은 평균법 적용에서 제외",
    "취득원가와 순실현가능가치 중 낮은 금액으로 측정",
    "순실현가능가치 = 추정 판매가격 - 적용 가능한 변동 판매비용",
    "저가법 평가 (판가 하락·완성원가 상승)",
    "과잉·진부화 평가 (재고 가치 하락)",
)
EXTRA_COLS = (
    "accounting_correct_1_or_0",
    "coverage_flags_6bits",
    "coverage_0_to_6",
    "review_method",
    "reviewed_at",
)


def load_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("평가 CSV의 열 이름이 없습니다.")
        return list(reader.fieldnames), list(reader)


def make_key(row):
    return str(row["condition"]).strip(), int(row["repeat"])


def atomic_save(path: Path, columns: list[str], rows: list[dict]):
    temp = path.with_name(path.name + ".tmp")
    try:
        with temp.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def valid_review(row):
    return (
        row.get("semantic_supported_1_or_0", "") in ("0", "1")
        and row.get("accounting_correct_1_or_0", "") in ("0", "1")
        and len(row.get("coverage_flags_6bits", "")) == 6
        and set(row["coverage_flags_6bits"]) <= {"0", "1"}
    )


def write_summary(folder: Path, rows: list[dict], by_key: dict):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get("condition", ""))].append(row)
    summary = []
    for label in "ABCD":
        members = groups.get(label, [])
        reviewed = [r for r in members if valid_review(r)]
        n = len(reviewed)
        auto = [by_key[make_key(r)] for r in members if make_key(r) in by_key]
        coverage = [r["coverage_flags_6bits"].count("1") for r in reviewed]
        summary.append({
            "condition": label,
            "total_runs": len(members),
            "manually_reviewed_runs": n,
            "auto_format_pass_count": sum(bool(r.get("success")) for r in auto),
            "groundedness_pass_count": sum(int(r["semantic_supported_1_or_0"]) for r in reviewed),
            "accounting_correct_pass_count": sum(int(r["accounting_correct_1_or_0"]) for r in reviewed),
            "mean_coverage_0_to_6": round(sum(coverage) / n, 2) if n else "",
            "mean_coverage_percent": round(100 * sum(coverage) / (6 * n), 1) if n else "",
        })
    path = folder / "answer_quality_summary.csv"
    atomic_save(path, list(summary[0].keys()), summary)
    print("\n=== 수동 답변 품질 평가 요약 ===")
    for r in summary:
        n = r["manually_reviewed_runs"]
        if n:
            print(f"{r['condition']}: 평가 {n}/{r['total_runs']}건, "
                  f"원문 충실 {r['groundedness_pass_count']}/{n}, "
                  f"회계 정확 {r['accounting_correct_pass_count']}/{n}, "
                  f"정보 포괄성 평균 {r['mean_coverage_0_to_6']}/6 "
                  f"({r['mean_coverage_percent']}%)")
        else:
            print(f"{r['condition']}: 아직 수동 평가 없음")
    print("[품질 요약 저장]", path)
    print("[주의] 단일 질문·소표본에 대한 수동 검토이며, 독립 전문가 평가가 아닙니다.")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Experiment 06 answer-quality review (manual)")
    parser.add_argument("--dir", required=True, type=Path, help="runs.json, metadata.json과 평가 CSV가 있는 실험 디렉터리")
    parser.add_argument("--summary-only", action="store_true", help="기존 평가 내용을 집계만 함")
    args = parser.parse_args(argv)
    folder = args.dir.expanduser().resolve()
    target = folder / "answer_quality_review.csv"
    runs_file = folder / "runs.json"
    meta_file = folder / "metadata.json"
    if not target.is_file():
        parser.error(f"파일이 없습니다: {target} (먼저 human_review.csv를 이름 변경했는지 확인하세요)")
    if not runs_file.is_file() or not meta_file.is_file():
        parser.error("실험 폴더에 runs.json과 metadata.json이 필요합니다.")

    cols, rows = load_csv(target)
    if not all(k in cols for k in ("condition", "repeat", "answer", "semantic_supported_1_or_0", "review_notes")):
        parser.error("기존 평가 CSV에 필요한 열이 없습니다. 열 이름을 확인하세요.")
    if len({make_key(r) for r in rows}) != len(rows):
        parser.error("(조건, 반복번호)가 중복되어 평가 결과를 연결할 수 없습니다.")
    original_runs = json.loads(runs_file.read_text(encoding="utf-8"))
    metadata = json.loads(meta_file.read_text(encoding="utf-8"))
    by_key = {make_key(r): r for r in original_runs}
    for row in rows:
        key = make_key(row)
        if key not in by_key:
            parser.error(f"runs.json에 존재하지 않는 평가 행: {key}")
        raw_answer = str((by_key[key].get("response") or {}).get("answer", ""))
        if row["answer"].strip() != raw_answer.strip():
            parser.error(f"평가 CSV와 runs.json의 답변이 다릅니다: {key}. 임의 평가를 방지하기 위해 중단합니다.")
    for col in EXTRA_COLS:
        if col not in cols:
            cols.append(col)
    for row in rows:
        for col in EXTRA_COLS:
            row.setdefault(col, "")

    if args.summary_only:
        write_summary(folder, rows, by_key)
        return 0

    print("\n[질문]", metadata.get("question", "(질문 메타데이터 없음)"))
    print("[공시 출처]", metadata.get("source"), metadata.get("company"), metadata.get("year"), metadata.get("basis"))
    print("\n[핵심 정보 6개: 원문에 있는 항목을 답변이 명시했을 때만 1]")
    for i, c in enumerate(CRITERIA, 1):
        print(f"  {i}. {c}")
    print("\n[참고 원문 - 상위 검색 결과]")
    for src in metadata.get("retrieved_passages", [])[:3]:
        print(f"{src.get('alias')} [{src.get('basis_context', '?')}] {src.get('section')} | {src.get('url') or '(가상 문서)'}")
        excerpt = str(src.get("excerpt", "")).replace("\n", " ")
        print("  " + excerpt[:540])
    print("\n[입력 방법] '원문충실성 회계정확성 6자리포괄성' (예: 1 1 100111)")
    print("  원문충실성: 1=답변 내용이 인용 원문으로 뒷받침됨 / 0=근거 밖 주장·왜곡 있음")
    print("  회계정확성: 1=회계적으로 확인 가능한 오류 없음 / 0=오류 있음")
    print("  6자리포괄성: 위 1~6번 항목을 포함했으면 1, 아니면 0")
    print("  s=해당 행 건너뛰기, q=평가 종료 및 중간 결과 저장")

    backup = target.with_name("answer_quality_review.pre_review_backup.csv")
    if not backup.exists():
        shutil.copy2(target, backup)
        print("[원본 백업]", backup)
    changed = False
    for row in sorted(rows, key=lambda r: (str(r["condition"]), int(r["repeat"]))):
        if valid_review(row):
            continue
        original = by_key[make_key(row)]
        response = original.get("response") or {}
        print(f"\n--- 조건 {row['condition']} / 반복 {row['repeat']} ---")
        print("[자동 검증]", "PASS" if original.get("success") else f"FAIL: {original.get('error_type')}")
        if response.get("status"):
            print("[모델 상태 필드]", response["status"])
        print("[인용]", original.get("citations", []))
        print("[답변]", row.get("answer", "") or "(답변 없음)")
        while True:
            try:
                entry = input("평가 입력 (예: 1 1 100111 / s / q): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n[중단] 입력된 평가만 보관합니다.")
                entry = "q"
            if entry in ("q", "quit"):
                write_summary(folder, rows, by_key)
                return 0
            if entry in ("s", "skip"):
                break
            parts = entry.split()
            if len(parts) == 3 and parts[0] in ("0", "1") and parts[1] in ("0", "1") and len(parts[2]) == 6 and set(parts[2]) <= {"0", "1"}:
                try:
                    note = input("평가 메모 (없으면 Enter): ").strip()
                except (EOFError, KeyboardInterrupt):
                    note = ""
                row["semantic_supported_1_or_0"] = parts[0]
                row["accounting_correct_1_or_0"] = parts[1]
                row["coverage_flags_6bits"] = parts[2]
                row["coverage_0_to_6"] = str(parts[2].count("1"))
                row["review_method"] = "manual_rubric_single_reviewer"
                row["reviewed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
                row["review_notes"] = note
                atomic_save(target, cols, rows)
                changed = True
                print(f"[저장 완료] 포괄성 {row['coverage_0_to_6']}/6")
                break
            print("입력 형식이 맞지 않습니다. 예: 1 1 100111")
    if not changed:
        print("\n[안내] 새로 입력된 평가가 없습니다.")
    write_summary(folder, rows, by_key)
    print("[평가 CSV]", target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
