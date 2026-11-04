import hashlib
import json
from pathlib import Path


def test_published_experiment_inputs_match_manifest():
    root = Path(__file__).resolve().parents[1] / "LLM_test/fixtures"
    manifest = json.loads((root / "manifest.json").read_text())
    for row in manifest["files"]:
        assert hashlib.sha256((root / row["file"]).read_bytes()).hexdigest() == row["sha256"]


def test_frozen_selector_cases_have_candidates_and_sources():
    import csv
    root = Path(__file__).resolve().parents[1] / "LLM_test/fixtures/exp08d/baseline"
    attempts = [json.loads(line) for line in (root / "attempts.jsonl").read_text().splitlines()]
    def rows(name):
        with (root / name).open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    candidates = rows("selector_candidates.csv")
    evidence = rows("retrieved_evidence.csv")
    cases = [a for a in attempts if a["condition"] == "hybrid_explicit_cfs"]
    assert len(cases) == 12
    for case in cases:
        chosen = [c for c in candidates if c["question_id"] == case["question_id"] and c["condition"] == case["condition"]]
        assert [c["candidate_id"] for c in chosen] == case["candidate_ids_offered"]
        assert any(e["question_id"] == case["question_id"] and e["condition"] == case["condition"] for e in evidence)
