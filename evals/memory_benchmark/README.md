# Hexis Public Memory Benchmark v1

This is a public, vendor-neutral benchmark for memory behavior across time. It has
25 synthetic cases—five each for provenance accuracy, contradiction detection,
recall after six months, cross-session continuity, and resistance to stale beliefs.
The corpus, schemas, adapters, scorer, and Hexis's imperfect first result ship
together.

The benchmark uses exact answer strings and evidence event IDs. It has no model
judge, hidden prompt, or subjective rubric. That makes a result cheap to reproduce
and hard to inflate with a favorable evaluator, although an open corpus can of
course be trained against. Treat v1 as a small public contract, not a complete
measure of memory or intelligence.

## Files

| File | Purpose |
|---|---|
| `cases.v1.jsonl` | 25 timestamped, session-scoped cases with public gold labels |
| `case.schema.v1.json` | Dataset case schema |
| `adapter-input.schema.v1.json` | Gold-free document sent to an agent wrapper |
| `prediction.schema.v1.json` | Wrapper response schema |
| `model.py` | Strict chronology, reference, and corpus-integrity validation |
| `scoring.py` | Deterministic case and macro scoring |
| `adapters.py` | Hexis, external-command, and two labeled sanity baselines |
| `results/2026-08-28.json` | First published comparison, profile, and failure disclosure |
| `results/2026-08-28-hexis-predictions.json` | Exact one-shot Hexis prediction submission behind the published score |

The built-in corpus SHA-256 is
`f92bd4dc54a5209cadd2af90706be163d539d4d7f455c068695b5a8dbb323149`.
Changing the corpus requires a deliberate version and hash update; the validator
refuses silent drift.

## Scoring

Each case asks for a natural-language `answer`, supporting `citations` (event IDs),
detected `contradictions` (event IDs), and an explicit `abstained` flag. Expected
answer phrases must occur as complete normalized tokens. Forbidden stale/distractor
phrases reduce answer credit. Citation and contradiction sets use exact F1.

| Dimension | Case score |
|---|---|
| Provenance accuracy | 50% answer correctness + 50% citation F1 |
| Contradiction detection | 50% answer correctness + 50% contradiction-event F1 |
| Six-month recall | answer correctness, including distractor penalty |
| Cross-session continuity | answer correctness, including distractor penalty |
| Stale-belief resistance | 50% current-answer recall + 50% absence of forbidden stale values |

The overall score is the macro average of the five dimension means, so adding cases
to one dimension cannot make it dominate the result. Missing or abstained cases score
zero. Submitting unknown or duplicate case IDs is an error.

## Run it

From a Hexis checkout and virtual environment:

```bash
python -m evals.memory_benchmark.run validate
python -m evals.memory_benchmark.run run --adapter append-only
python -m evals.memory_benchmark.run run --adapter recent-window
python -m evals.memory_benchmark.run run --adapter hexis --live-contradictions
```

The Hexis adapter uses the configured database and embedding service. Every case is
created, queried, and revised inside its own transaction, and the transaction is
always rolled back. Only case-owned memory IDs can become evidence. The optional
live contradiction pass uses Hexis's configured subconscious/heartbeat model and
the production contradiction prompt; omitting it is valid but truthfully measures
that dimension with the detector disabled.

By default results go to `${XDG_CACHE_HOME:-~/.cache}/hexis/memory-benchmark/`, not
the source tree. Pass `--output` only when intentionally publishing an artifact.

## Run another agent

Write a wrapper that reads one JSON object from stdin and writes one prediction JSON
object to stdout. The input contains all timestamped events and the query but omits
the `expected` block. The process is invoked once per case, so the wrapper must reset
or namespace the agent under test. Then run:

```bash
python -m evals.memory_benchmark.run run \
  --adapter command \
  --name your-agent-and-version \
  --command './your-wrapper --json' \
  --output result.json
```

Commands are parsed to argv and executed without a shell. The wrapper should feed
events in chronological order, honor session boundaries and timestamps, ask the
query, and map its response/evidence into:

```json
{
  "case_id": "provenance-01",
  "answer": "The code is LARCH-742.",
  "citations": ["p01-user-note"],
  "contradictions": [],
  "abstained": false
}
```

Publish the wrapper, product/model version, memory settings, raw run result, corpus
hash, and any retries. Do not publish a score produced from hand-authored prediction
files as an agent run.

## First result—and where Hexis loses

The 2026-08-28 run used Hexis 1.0.13, its live DB-owned temporal/revision functions,
local `embeddinggemma:300m-qat-q4_0`, and the configured
`openai-codex/gpt-5.6-luna` contradiction classifier.

| System | Kind | Provenance | Contradiction | Six months | Cross-session | Stale beliefs | Overall |
|---|---|---:|---:|---:|---:|---:|---:|
| Hexis memory v1 | system under test | 96.67 | 85.00 | 100.00 | 100.00 | 100.00 | 96.33 |
| Append-only transcript | sanity baseline | 96.67 | 55.00 | 100.00 | 100.00 | 60.00 | 82.33 |
| Recent 30-day window | sanity baseline | 20.00 | 10.00 | 0.00 | 100.00 | 30.00 | 32.00 |

Hexis lost points on two cases:

- `provenance-02`: it answered `USD 48000` correctly but cited only one of two
  corroborating sources, for citation F1 66.67 and case score 83.33.
- `contradiction-03`: in the official one-shot run, the live production classifier
  emitted no conflict and deterministic answer projection surfaced only
  `ALMOND-SAFE`, not `SEVERE-ALMOND-ALLERGY`, for case score 25.

Those are not rounded away or rerun until green. They expose two real seams: evidence
aggregation and the handoff from detected conflict to the final answer.

An earlier development pilot scored 98.33 because the same configured classifier
did identify the allergen pair. The lower 96.33 result from the complete `run-all`
journey is the official result; there is no best-of retry. That observed variance is
itself a limitation of the live-model contradiction axis.

No third-party product result appears in the table because none was actually run in
this environment. The comparison rows are source-controlled baselines, explicitly
labeled as such. Submissions from other agents are welcome; unsupported marketing
numbers are not.

Re-score the published Hexis submission directly:

```bash
python -m evals.memory_benchmark.run score \
  evals/memory_benchmark/results/2026-08-28-hexis-predictions.json
```

## Limits of v1

- The corpus is small, synthetic, English, and public.
- Events are already normalized into memory-worthy statements; v1 does not score
  conversational fact extraction.
- Exact strings avoid judge bias but under-credit valid paraphrases that omit the
  benchmark token.
- The Hexis adapter measures the memory substrate plus the production contradiction
  classifier, with deterministic answer projection—not the full chat personality.
- It does not measure latency under a large personal corpus, privacy leakage, or
  adversarial prompt injection. Those need separate suites.

Please propose new cases by adding them to a new corpus version. Changing v1 gold or
wording in place invalidates its published hash and existing results.
