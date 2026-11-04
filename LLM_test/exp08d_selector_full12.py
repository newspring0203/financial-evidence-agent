"""Exp 08-D: frozen 12-question selector regression test.

Reuses v4 retrieval results and candidate passages.
Does not run retrieval or answer generation.
"""
import hashlib
import json
from datetime import datetime

import httpx

from LLM_test import exp08d_selector_probe as probe

BASE = probe.BASE
CONDITION = "hybrid_explicit_cfs"

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    attempts_file = BASE / "attempts.jsonl"
    candidates_file = BASE / "selector_candidates.csv"
    evidence_file = BASE / "retrieved_evidence.csv"
    metadata_file = BASE / "metadata.json"

    attempts = [
        json.loads(line)
        for line in attempts_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    candidates = probe.read_csv(candidates_file)
    evidence = probe.read_csv(evidence_file)
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    config = metadata["config"]

    cases = sorted(
        (r for r in attempts if r["condition"] == CONDITION),
        key=lambda r: r["question_id"]
    )

    assert len(cases) == 12, "12문항이 아닙니다."
    assert len({r["question_id"] for r in cases}) == 12

    gold_refs = {
        (r["question_id"], r["ref"])
        for r in evidence
        if r["condition"] == CONDITION and r["is_gold"] == "1"
    }

    output = BASE.parent / (
        "experiment08d_full12_"
        + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)

    model = config["generation_model"]
    predict = int(config["selector_num_predict"])
    results = []

    print("[Exp08-D] 12문항 Selector 회귀 평가")
    print("[모델]", model)
    print("[출력]", output, flush=True)

    with httpx.Client(timeout=240.0) as client, \
         (output / "results.jsonl").open(
             "w", encoding="utf-8"
         ) as log:

        for r in cases:
            qid = r["question_id"]
            rows = [
                c for c in candidates
                if c["question_id"] == qid
                and c["condition"] == CONDITION
            ]
            allowed = [c["candidate_id"] for c in rows]

            assert allowed, f"{qid}: 후보 없음"
            assert len(allowed) == len(set(allowed))
            assert allowed == r["candidate_ids_offered"], (
                f"{qid}: 기존 후보와 불일치"
            )

            schema = {
                "type": "object",
                "properties": {
                    "candidate_id": {
                        "type": "string",
                        "enum": ["NONE"] + allowed
                    }
                },
                "required": ["candidate_id"],
                "additionalProperties": False
            }

            passages = [
                {
                    "candidate_id": c["candidate_id"],
                    "section": c["section"],
                    "quote": c["support_quote"]
                }
                for c in rows
            ]

            messages = [
                {"role": "system", "content": probe.SYSTEM},
                {
                    "role": "user",
                    "content": (
                        "질문: " + r["question"]
                        + "\n\n근거 후보(JSON):\n"
                        + json.dumps(passages, ensure_ascii=False)
                    )
                }
            ]

            response = client.post(
                "http://127.0.0.1:11434/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "format": schema,
                    "think": False,
                    "options": {
                        "temperature": 0,
                        "seed": 42,
                        "num_predict": predict
                    }
                }
            )
            response.raise_for_status()
            api = response.json()
            raw = (api.get("message") or {}).get("content", "")

            selected = None
            error = ""
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("not_object")
                if set(parsed) != {"candidate_id"}:
                    raise ValueError("invalid_fields")
                selected = parsed["candidate_id"]
                if not isinstance(selected, str) or selected not in (
                    ["NONE"] + allowed
                ):
                    raise ValueError("invalid_candidate")
            except (ValueError, TypeError) as e:
                error = str(e)

            chosen = next(
                (c for c in rows if c["candidate_id"] == selected),
                None
            )

            gold = bool(
                chosen is not None
                and (qid, chosen["ref"]) in gold_refs
            )

            result = {
                "question_id": qid,
                "question": r["question"],
                "condition": CONDITION,
                "baseline_status": r["status"],
                "baseline_selection_error": r["selection_error"],
                "baseline_cited_gold": r["citation_contains_gold"],
                "candidate_id": selected,
                "selected_ref": chosen["ref"] if chosen else None,
                "selected_quote": (
                    chosen["support_quote"] if chosen else None
                ),
                "valid_output": not bool(error),
                "selection_error": error,
                "gold_chunk_selected": gold,
                "raw_response": raw
            }
            results.append(result)
            log.write(json.dumps(result, ensure_ascii=False) + "\n")
            log.flush()

            print(
                f"{qid}: v4={r['status']} "
                f"-> selected={selected} "
                f"/ valid={not bool(error)} "
                f"/ gold={gold}",
                flush=True
            )

    experiment_metadata = {
        "experiment": "exp08d_selector_full12",
        "baseline": str(BASE),
        "baseline_attempts_sha256": sha256(attempts_file),
        "baseline_candidates_sha256": sha256(candidates_file),
        "baseline_evidence_sha256": sha256(evidence_file),
        "model": model,
        "temperature": 0,
        "seed": 42,
        "num_predict": predict,
        "selector_prompt_sha256": hashlib.sha256(
            probe.SYSTEM.encode()
        ).hexdigest(),
        "note": (
            "Previously used evaluation data. Regression test only. "
            "Gold chunk selection is not answer correctness."
        )
    }

    (output / "metadata.json").write_text(
        json.dumps(experiment_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\n=== FULL 12 SUMMARY ===")
    print("문항:", len(results))
    print("형식 유효:", sum(r["valid_output"] for r in results))
    print("후보 선택:", sum(
        r["candidate_id"] not in (None, "NONE")
        and r["valid_output"]
        for r in results
    ))
    print("Gold 문단 선택:", sum(
        r["gold_chunk_selected"] for r in results
    ))
    print("NONE 선택:", sum(
        r["candidate_id"] == "NONE" for r in results
    ))
    print("[저장]", output)
    print("[주의] 최종 답변 정확도는 평가하지 않았습니다.")

if __name__ == "__main__":
    main()
