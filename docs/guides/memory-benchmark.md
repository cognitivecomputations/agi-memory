<!--
title: Public Memory Benchmark
summary: Reproduce Hexis's public memory score or run another agent on the same cases
read_when:
  - "You want evidence for Hexis memory claims"
  - "You want to compare another agent's long-term memory behavior"
  - "You want to reproduce the published memory benchmark result"
section: guides
-->

# Public Memory Benchmark

Hexis ships a public, deterministic memory benchmark under
`evals/memory_benchmark/`. Version 1 has 25 synthetic cases—five each for:

- provenance accuracy;
- contradiction detection;
- recall after more than six months;
- continuity across session boundaries; and
- resistance to explicitly superseded beliefs.

The corpus, JSON schemas, validator, scorer, external-agent protocol, raw adapter
code, and published result are all source-controlled. Scoring uses exact answer
phrases and evidence event IDs, not an LLM judge.

## Reproduce it

Activate the Hexis environment, make sure the database and embedding service are
healthy, then run:

```bash
python -m evals.memory_benchmark.run validate
python -m evals.memory_benchmark.run run --adapter hexis --live-contradictions
```

The Hexis adapter writes each case into one transaction, exercises the real memory,
temporal recall, and supersession functions, and always rolls back. It filters
evidence to case-owned memory IDs, so existing personal memory can neither help nor
hurt a score. `--live-contradictions` uses the currently configured
subconscious/heartbeat model and production detector prompt. Leave it off to test
the structural memory path without making model calls; the result will explicitly
say that detection was disabled.

Results default to `${XDG_CACHE_HOME:-~/.cache}/hexis/memory-benchmark/`. Nothing is
written into the checkout unless `--output` names a repository path intentionally.

## First published result

The 2026-08-28 run used Hexis 1.0.13, local
`embeddinggemma:300m-qat-q4_0`, and `openai-codex/gpt-5.6-luna` for the five
contradiction-classification cases.

| System | Provenance | Contradiction | Six months | Cross-session | Stale beliefs | Overall |
|---|---:|---:|---:|---:|---:|---:|
| Hexis memory v1 | 96.67 | 85.00 | 100.00 | 100.00 | 100.00 | 96.33 |
| Append-only transcript baseline | 96.67 | 55.00 | 100.00 | 100.00 | 60.00 | 82.33 |
| Recent 30-day window baseline | 20.00 | 10.00 | 0.00 | 100.00 | 30.00 | 32.00 |

The two comparison rows are source-controlled sanity baselines, not third-party
products. No competitor result is claimed because no competitor was actually run in
this environment.

Hexis also does not claim a perfect run. It lost evidence credit when a correct
two-source answer cited only one corroborating source. In a safety-critical allergen
case, the live classifier emitted no conflict and the deterministic answer projection
surfaced only one side. Those failures remain in the published artifact because they
identify real work: evidence aggregation and detector-to-answer handoff.

An earlier development pilot scored 98.33 after detecting that allergen pair. The
lower 96.33 one-shot `run-all` result is official; Hexis does not select the best run.
This variance is itself a disclosed limitation of the live-model contradiction axis.

## Run another agent

An external wrapper reads one gold-free JSON case from stdin and writes one
prediction object to stdout. The runner invokes it once per case:

```bash
python -m evals.memory_benchmark.run run \
  --adapter command \
  --name your-agent-and-version \
  --command './your-wrapper --json' \
  --output result.json
```

Publish the wrapper, exact product/model and memory settings, raw result, corpus
SHA-256, and retry policy. The complete protocol, score formula, current corpus hash,
known limitations, and failure-level result are in
[`evals/memory_benchmark/README.md`](https://github.com/QuixiAI/Hexis/blob/main/evals/memory_benchmark/README.md).

## What v1 does not prove

The cases are small, synthetic, English, and public. Statements arrive already
normalized as memory-worthy events, so the suite does not test conversational fact
extraction. Exact tokens make scoring reproducible but can under-credit a good
paraphrase. It also does not measure privacy leakage, large-corpus latency, or prompt
injection. Results should be read as one narrow behavioral measurement, not a claim
of general intelligence or universal memory quality.
