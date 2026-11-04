"""Test real local LLM tool calling against Financial Evidence Agent domain tools.

Run from the repository root:
    python local_ollama_test.py
    python local_ollama_test.py --live --company 삼성전자 --year 2024

Only a locally installed Ollama model is used. No OpenAI API credits needed.
"""

import argparse
import json
from getpass import getpass

from ollama import Client

from fea.agent import Run
from fea.domain import TOOL_DEFINITIONS

MODEL = "qwen3:1.7b"


def sample_data():
    """Synthetic fixture: demonstrates real LLM reasoning/tool choice, not real filings."""
    code = "00000000"
    return {
        "companies": [{"name": "샘플기업", "corp_code": code}],
        "facts": [
            {
                "id": "demo:revenue:2025",
                "company": code,
                "year": 2025,
                "basis": "CFS",
                "metric": "revenue",
                "amount": "120000",
                "unit": "원",
                "currency": "KRW",
                "report_code": "11011",
            },
            {
                "id": "demo:revenue:2024",
                "company": code,
                "year": 2024,
                "basis": "CFS",
                "metric": "revenue",
                "amount": "100000",
                "unit": "원",
                "currency": "KRW",
                "report_code": "11011",
            },
        ],
        "documents": [
            {
                "id": "demo:report:2025",
                "title": "샘플기업 2025년 사업보고서 (가상 자료)",
                "company": code,
                "year": 2025,
                "basis": "CFS",
                "text": "1. 재무제표 주석\n샘플기업은 매출액을 원화 기준으로 표시합니다.\n",
                "synthetic": True,
            }
        ],
    }


def make_ollama_tools():
    # Repository uses OpenAI's flat function schema; Ollama nests it.
    return [
        {
            "type": "function",
            "function": {
                "name": item["name"],
                "description": item["description"],
                "parameters": item["parameters"],
            },
        }
        for item in TOOL_DEFINITIONS
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Use real OpenDART filings")
    parser.add_argument("--company", default="삼성전자", help="Company name for OpenDART")
    parser.add_argument("--year", type=int, default=2024, help="Current fiscal year")
    parser.add_argument("--model", default=MODEL, help="Installed Ollama model tag")
    args = parser.parse_args()

    if args.live:
        from fea.live import load_public

        dart_key = getpass("OpenDART API 키 (입력 내용 숨김): ").strip()
        if not dart_key:
            raise SystemExit("OpenDART 키가 필요합니다. 키가 없다면 --live 없이 실행하세요.")
        print("실제 공시를 수집하는 중입니다. (LLM 호출과는 별개)")
        data = load_public(dart_key, args.company, args.year, "CFS")
        name, year, source = args.company, args.year, "dart"
    else:
        data = sample_data()
        name, year, source = "샘플기업", 2025, "demo"
        print("[안내] 가상 재무 수치 기반 테스트 | AI 모델: Qwen3 1.7B (Ollama 로컬 실행)")

    run = Run(name, year, "CFS", source, "bm25", data=data)
    tools = make_ollama_tools()
    messages = [
        {
            "role": "system",
            "content": (
                "당신은 재무 공시 도구를 사용하는 한국어 AI Agent입니다. "
                "금액을 지어내지 마세요. 반드시 제공된 함수를 호출해 당기와 전기 매출액을 "
                "각각 조회한 뒤 calculate_change로 증감률을 계산하세요. "
                "도구 호출의 company 인수는 정확히 " + run.company + " 입니다. "
                "year는 정수, basis는 CFS를 쓰세요. "
                "계산 결과와 사용한 도구, 확인된 출처 ID를 간결하게 알려주세요. "
                "도구 실행에 실패하면 실패했다고 말하세요."
            ),
        },
        {
            "role": "user",
            "content": f"{name} {year}년과 {year-1}년 연결 매출액을 조회하고 증감률을 계산해 줘.",
        },
    ]

    client = Client(host="http://127.0.0.1:11434", timeout=120)
    ok_calls = 0
    for turn in range(8):
        print(f"\n=== LLM 응답 {turn + 1} ===")
        reply = client.chat(
            model=args.model,
            messages=messages,
            tools=tools,
            think=False,
            stream=False,
            options={"temperature": 0, "num_ctx": 4096},
        )
        messages.append(reply.message)
        calls = reply.message.tool_calls or []
        if not calls:
            print("[최종 답변]", reply.message.content or "(응답 텍스트 없음)")
            break

        for call in calls:
            tool_name = call.function.name
            tool_args = call.function.arguments
            print("[LLM이 선택한 함수]", tool_name)
            print("[LLM이 만든 인수]", json.dumps(tool_args, ensure_ascii=False))
            result = run.call(tool_name, tool_args)
            success = not (isinstance(result, dict) and "error" in result)
            ok_calls += int(success)
            print("[Python 도구 실행 결과]", json.dumps(result, ensure_ascii=False)[:1200])
            messages.append({
                "role": "tool",
                "tool_name": tool_name,
                "content": json.dumps(result, ensure_ascii=False),
            })
    else:
        print("[중단] 최대 8회의 LLM 응답 단계에 도달했습니다.")

    print("\n=== 검증 요약 ===")
    print(f"정상 도구 호출: {ok_calls}회")
    print("실행된 도구:", [item["tool"] for item in run.trace])
    print("계산 도구 호출 여부:", any(item["tool"] == "calculate_change" and item["status"] == "ok" for item in run.trace))
    print("주의: 도구 실행 성공은 최종 답변의 회계적 타당성까지 보장하지 않습니다.")


if __name__ == "__main__":
    main()
