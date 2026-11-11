# E09 evaluation tools

Run from the repository root with `PYTHONPATH=.`. These tools reuse the existing
E08-D pipeline without changing its selector, prompts, or generation behavior.
Local Gold CSVs, question drafts, sealed holdout, snapshots, cached vectors,
model logs and human review results are intentionally excluded from this repository.
The question generator is also excluded because it embeds evaluation questions.

- `validate_gold.py`: annotation schema, quote hashes and optional source offsets.
- `prepare_e09_candidates.py`: independent navigation hints for human review;
  these are not the evaluated retriever and never approve Gold.
- `verify_gold_against_local_snapshot.py`: source, chunk, CFS and offset checks.
- `review_gold_interactive.py`: record explicit human Gold approvals separately.
- `run_e09_frozen.py`: run verified Samsung 2024 CFS note questions with local
  Ollama, the original dataset/scope audit and cached embeddings.
- `grade_e09.py`: independently aggregate Hit@3, All Gold Hit@3 and human grades.

Use `PYTHONPATH=. python E09_package/<tool>.py --help` for required local inputs.
The runner additionally needs the experiment dependencies in
`requirements-experiments.txt`, Ollama models `qwen3:1.7b` and
`qwen3-embedding:4b`, and the original local corpus artifacts. Model digests
must be recorded separately; model names alone do not guarantee reproducibility.

Only `verified` Gold is scored. Holdout is excluded by default and requires an
explicit runner/grader flag; regression is a separate opt-in grading split.
Mechanical source checks do not establish semantic correctness. E09-36 remains
unapproved and outside E09-01 scoring; checking related notes and table scope
is necessary before reconsidering approval. No historical scores are generated
or claimed by this source-only package.

Offline tests (synthetic data only):

```sh
python -m unittest discover -s E09_package/tests -v
```
