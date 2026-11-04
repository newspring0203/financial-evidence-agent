"""Experiment 03: Qwen3 local Function Calling + structured output validation.

Place this file in financial-evidence-agent/LLM_test/ and run from repo root:
    python -m LLM_test.local_ollama_test_v3
    python -m LLM_test.local_ollama_test_v3 --live --company 삼성전자 --year 2024

Requires the existing LLM_test/local_ollama_test_v2.py, the repository's fea
package, and locally running Ollama with qwen3:1.7b installed.
This does not use OpenAI API credits or modify the production Agent.
"""

import argparse
import json
import re
import time
from decimal import Decimal, InvalidOperation
from getpass import getpass

from jsonschema import validate, ValidationError
from ollama import Client, ResponseError

from fea.agent import Run
from LLM_test.exp02_tool_call_retry import sample_data, make_ollama_tools

MODEL = "qwen3:1.7b"

# A deliberately small schema for a 1.7B local model. The Python validator
# independently checks the values and IDs against actual tool results.
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "change_percent": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["change_percent", "evidence_ids", "summary"],
    "additionalProperties": False,
}


def successful_records(run, year):
    """Return facts and a calculation only after the shared domain tools ran."""
    facts = [r for r in run.retrieved if r.get("metric") == "revenue"
             and r.get("basis") == "CFS" and r.get("company") == run.company]
    current = next((r for r in facts if r.get("year") == year), None)
    prior = next((r for r in facts if r.get("year") == year - 1), None)
    if not current or not prior:
        return current, prior, None
    calculated_ids = {
        entry["result"].get("id")
        for entry in run.trace
        if entry["tool"] == "calculate_change" and entry["status"] == "ok"
        and isinstance(entry.get("result"), dict)
    }
    calc = next((r for r in run.retrieved
                 if r.get("kind") == "calculation" and r.get("id") in calculated_ids
                 and r.get("current_id") == current["id"]
                 and r.get("prior_id") == prior["id"]), None)
    return current, prior, calc


def check_final(output, current, prior, calc, run):
    """Structural, numeric and evidence-integrity checks; not semantic auditing."""
    validate(instance=output, schema=OUTPUT_SCHEMA)
    if not output["summary"].strip():
        raise ValueError("요약 문장이 비어 있습니다.")
    if re.search(r"</?\s*(?:think|tool_response)\b|\bfunction_call\b", output["summary"], re.I):
        raise ValueError("최종 답변에 내부 추론 또는 도구 응답 표시가 섞여 있습니다.")
    try:
        actual = Decimal(output["change_percent"].replace("%", "").strip())
        expected = Decimal(calc["value"])
    except (InvalidOperation, AttributeError):
        raise ValueError("증감률은 숫자 문자열이어야 합니다.") from None
    if not actual.is_finite() or actual != expected:
        raise ValueError(f"증감률 불일치: LLM={output['change_percent']}, 계산 도구={calc['value']}")

    expected_ids = {current["id"], prior["id"], calc["id"]}
    actual_ids = output["evidence_ids"]
    if len(actual_ids) != len(expected_ids) or set(actual_ids) != expected_ids:
        raise ValueError("출처 ID 누락·중복·추가: 필요한 ID 3개와 정확히 일치해야 합니다.")

    # Cross-check against the existing repo's evidence verification logic.
    checks = run.tools.verify([current, prior, calc], run.retrieved)
    if len(checks) != 3 or not all(x.get("valid") for x in checks):
        raise ValueError("기존 Financial Evidence Agent의 근거 무결성 검사 실패")

    # Prevent an obvious contradictory percentage in model-written prose.
    percentages = re.findall(r"(?<![\d.])(-?\d+(?:\.\d+)?)\s*%", output["summary"])
    if any(Decimal(p) != expected for p in percentages):
        raise ValueError("요약 문장 속 퍼센트가 계산 도구 결과와 다릅니다.")
    return checks


def invoke_tools(client, model, run, company_name, year):
    tools = make_ollama_tools()
    messages = [
        {"role": "system", "content": (
            "당신은 재무 조회 도구를 사용하는 한국어 Agent입니다. "
            "선택 기업의 corp_code는 " + run.company + "이고 basis는 CFS입니다. "
            "get_financials를 당기·전기 각각 실행하고 나서, 반드시 그 결과의 id로 "
            "calculate_change를 실제 함수 호출하세요. 숫자를 직접 계산하지 마세요. "
            "도구 호출 이후에는 간단히 끝내세요."
        )},
        {"role": "user", "content": (
            f"{company_name}의 {year}년과 {year-1}년 연결 매출액을 조회하고 "
            "calculate_change 도구로 증감률을 계산해 주세요."
        )},
    ]
    corrections = 0
    for step in range(8):
        current, prior, calc = successful_records(run, year)
        if calc is not None:
            return current, prior, calc, corrections

        print(f"\n=== 도구 호출 단계 {step + 1} ===")
        response = client.chat(
            model=model, messages=messages, tools=tools, think=False, stream=False,
            options={"temperature": 0, "num_ctx": 4096},
        )
        messages.append(response.message)
        calls = response.message.tool_calls or []
        if not calls:
            current, prior, _ = successful_records(run, year)
            if corrections >= 2:
                break
            corrections += 1
            if current and prior:
                repair = ("아직 calculate_change의 실제 실행 기록이 없습니다. "
                          f"반드시 calculate_change(current_id='{current['id']}', "
                          f"prior_id='{prior['id']}')를 도구로 호출하세요.")
            else:
                repair = ("매출 수치 두 개의 실제 조회가 아직 완료되지 않았습니다. "
                          f"get_financials로 {year}년 및 {year-1}년 매출을 각각 조회하세요. "
                          "그 후 calculate_change를 호출하세요.")
            print(f"[검증 게이트] 필수 도구 실행 미완료 → 재요청 {corrections}/2")
            messages.append({"role": "user", "content": repair})
            continue

        for call in calls:
            name = call.function.name
            arguments = call.function.arguments
            print("[Qwen이 선택한 도구]", name)
            print("[Qwen이 만든 인수]", json.dumps(arguments, ensure_ascii=False))
            result = run.call(name, arguments)
            print("[Python 도구 결과]", json.dumps(result, ensure_ascii=False)[:1000])
            messages.append({"role": "tool", "tool_name": name,
                             "content": json.dumps(result, ensure_ascii=False)})

    current, prior, calc = successful_records(run, year)
    return current, prior, calc, corrections


def structured_response(client, model, company_name, year, current, prior, calc, run):
    evidence = [current, prior, calc]
    # Give only verified tool results to a separate formatting call. Do not
    # reuse raw previous model prose, which may contain tool or think tags.
    source_payload = {
        "company": company_name,
        "year": year,
        "basis": "CFS",
        "verified_tool_results": evidence,
    }
    messages = [
        {"role": "system", "content": (
            "당신은 재무 분석 도구의 검증된 결과를 짧은 한국어 JSON 보고서로 정리합니다. "
            "주어진 도구 결과만 사용하세요. 추가 계산·원인 추정은 하지 마세요. "
            "change_percent에는 calculation.value와 동일한 숫자 문자열만 넣으세요. "
            "evidence_ids에는 현재 매출, 전기 매출, 계산 결과의 id를 모두 넣으세요. "
            "summary는 증감률을 한 문장으로 설명하세요. "
            "내부 추론, <think>, <tool_response>, 마크다운은 출력하지 마세요. "
            "반드시 JSON Schema를 따르세요."
        )},
        {"role": "user", "content": json.dumps(source_payload, ensure_ascii=False)},
    ]
    for attempt in range(1, 4):
        print(f"\n=== 구조화 출력 검증 {attempt}/3 ===")
        response = client.chat(
            model=model, messages=messages, format=OUTPUT_SCHEMA,
            think=False, stream=False, options={"temperature": 0, "num_ctx": 4096},
        )
        raw = response.message.content or ""
        try:
            parsed = json.loads(raw)
            checks = check_final(parsed, current, prior, calc, run)
            print("[JSON Schema] 통과")
            print("[증감률 대사] 통과")
            print("[출처 ID / 무결성] 통과")
            print("[최종 구조화 출력]")
            print(json.dumps(parsed, ensure_ascii=False, indent=2))
            print("[근거 검사]", json.dumps(checks, ensure_ascii=False))
            print("[주의] 자연어 요약의 전체 회계적 타당성은 별도 전문가 검토가 필요합니다.")
            return True
        except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as error:
            print(f"[검증 실패] {str(error)[:220]}")
            # Do not treat the previous malformed response as a correct final.
            # Send a concise repair instruction; no fake function calls.
            messages.append({"role": "assistant", "content": raw[:1000]})
            messages.append({"role": "user", "content": (
                "이전 출력은 검증 실패했습니다: " + str(error)[:180] + ". "
                f"change_percent는 정확히 '{calc['value']}', "
                "evidence_ids에는 아래 3개 ID만 중복 없이 사용하세요: "
                + json.dumps([r["id"] for r in evidence], ensure_ascii=False)
                + ". JSON 객체만 다시 작성하세요."
            )})
    print("[최종 결과] 구조화 출력 검증 실패 — 미검증 답변을 성공으로 처리하지 않습니다.")
    return False


def main():
    parser = argparse.ArgumentParser(description="Local Qwen tool-calling + JSON schema experiment")
    parser.add_argument("--live", action="store_true", help="Use OpenDART public filings")
    parser.add_argument("--company", default="삼성전자")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    if args.live:
        from fea.live import load_public
        key = getpass("OpenDART API 키 (입력 표시 안 됨): ").strip()
        if not key:
            raise SystemExit("OpenDART 키가 필요합니다.")
        print("[데이터] 실제 OpenDART 공시 수집")
        data = load_public(key, args.company, args.year, "CFS")
        company_name, year, source = args.company, args.year, "dart"
    else:
        print(f"[모델] {args.model} · Ollama 로컬 실행")
        print("[데이터] 가상 재무 수치 (LLM 함수 호출 자체는 실제 실행)")
        data = sample_data()
        company_name, year, source = "샘플기업", 2025, "demo"

    run = Run(company_name, year, "CFS", source, "bm25", data=data)
    client = Client(host="http://127.0.0.1:11434", timeout=180)
    started = time.perf_counter()
    try:
        current, prior, calc, corrections = invoke_tools(client, args.model, run, company_name, year)
        print("\n=== 도구 호출 평가 ===")
        print("실행된 도구:", [e["tool"] for e in run.trace])
        print("재요청 횟수:", corrections)
        if calc is None:
            print("[실패] 당기·전기 조회 및 계산 도구 호출이 완료되지 않았습니다.")
            raise SystemExit(1)
        print("[통과] 재무 조회 2회 + calculate_change 호출")
        passed = structured_response(client, args.model, company_name, year, current, prior, calc, run)
        print(f"\n[실험 요약] 모델={args.model} / 도구 호출=통과 / JSON·수치·근거 검증={'통과' if passed else '실패'}")
        print(f"소요 시간: {time.perf_counter() - started:.1f}초")
        if not passed:
            raise SystemExit(1)
    except (ResponseError, ConnectionError, OSError) as error:
        raise SystemExit("Ollama 연결/실행 오류: " + str(error)) from error


if __name__ == "__main__":
    main()
