"""Ask the model about N things in one call, not N calls.

A loop that calls the model once per item treats a batch problem as a per-row
problem. `services/connector_cognition.py` did exactly that over a query with
`LIMIT 80`: eighty round trips, eighty prompt preambles, eighty independent
chances to fail, for work the model will happily do in one request.

This is a shared helper rather than a central queue. Batching is only sound
among *like* requests — mixing a summarization prompt with an importance
classification in one call degrades both — so each caller supplies its own
system prompt and schema and hands over its items. A queue would have to
sub-divide by (prompt, schema, model) to be correct, which is this, with extra
indirection. See PLAN.md §13.3·B2 for the full argument.

What this owns, so no caller reimplements it:

* results keyed by the item's own id, never by array position
* chunking by size, so a batch never silently overflows the model's context
* per-chunk failure, so one bad response does not lose the whole run
* one shared context block sent once per chunk rather than once per item
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Iterable, Sequence, TypeVar

from core.llm_json import chat_json

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

# Chunk budget in characters. A character count rather than a token count on
# purpose: it needs no tokenizer, no per-provider table, and no network call,
# and ~4 chars/token is close enough for a budget whose only job is to stay
# clear of the context limit. Deliberately conservative.
DEFAULT_CHUNK_CHARS = 24_000

# Below this there is nothing to batch, and a one-item "batch" only adds a
# wrapper the model has to reason about.
MIN_BATCH_SIZE = 2


def chunk_by_size(
    items: Sequence[T],
    *,
    size_of: Callable[[T], int],
    budget: int = DEFAULT_CHUNK_CHARS,
) -> list[list[T]]:
    """Split items into chunks that each fit the budget.

    Eighty short Slack messages fit in one call; eighty long emails do not, so
    chunking by count would either waste calls or overflow. An item larger than
    the whole budget still gets its own chunk — truncating it silently would be
    worse than one oversized request the model can refuse.
    """
    chunks: list[list[T]] = []
    current: list[T] = []
    used = 0
    for item in items:
        cost = max(size_of(item), 1)
        if current and used + cost > budget:
            chunks.append(current)
            current = []
            used = 0
        current.append(item)
        used += cost
    if current:
        chunks.append(current)
    return chunks


async def batch_classify(
    items: Sequence[T],
    *,
    llm_config: dict[str, Any],
    system: str,
    key: Callable[[T], str],
    item_payload: Callable[[T], dict[str, Any]],
    parse: Callable[[Any], R],
    fallback: Callable[[T], R],
    shared: dict[str, Any] | None = None,
    output_hint: Any = None,
    max_tokens: int = 4000,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    temperature: float = 0.1,
    provenance: dict[str, str] | None = None,
) -> dict[str, R]:
    """Classify every item, in as few model calls as the budget allows.

    Returns a verdict per item keyed by ``key(item)``. Every item is present in
    the result: anything the model did not answer for falls back, so callers
    never have to distinguish "missing" from "empty".

    ``shared`` is context identical for every item — existing claims, an output
    schema — sent once per chunk instead of once per item.
    """
    results: dict[str, R] = {}
    if not items:
        return results

    keyed: list[tuple[str, T]] = []
    seen: set[str] = set()
    for item in items:
        item_key = str(key(item))
        if item_key in seen:
            # A duplicate inside one batch is the same question twice.
            continue
        seen.add(item_key)
        keyed.append((item_key, item))

    def _size(pair: tuple[str, T]) -> int:
        try:
            return len(json.dumps(item_payload(pair[1]), ensure_ascii=False))
        except Exception:
            return chunk_chars  # unserializable: give it a chunk of its own

    for chunk in chunk_by_size(keyed, size_of=_size, budget=chunk_chars):
        verdicts = await _classify_chunk(
            chunk,
            llm_config=llm_config,
            system=system,
            item_payload=item_payload,
            shared=shared,
            output_hint=output_hint,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        for item_key, item in chunk:
            raw = verdicts.get(item_key)
            if raw is None:
                results[item_key] = fallback(item)
                if provenance is not None:
                    provenance[item_key] = "fallback"
                continue
            try:
                results[item_key] = parse(raw)
                if provenance is not None:
                    provenance[item_key] = "llm"
            except Exception:
                logger.debug(
                    "Batch verdict unparseable for %s", item_key, exc_info=True
                )
                results[item_key] = fallback(item)
                if provenance is not None:
                    provenance[item_key] = "fallback"

    return results


async def _classify_chunk(
    chunk: Sequence[tuple[str, Any]],
    *,
    llm_config: dict[str, Any],
    system: str,
    item_payload: Callable[[Any], dict[str, Any]],
    shared: dict[str, Any] | None,
    output_hint: Any,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    """One model call for one chunk. Returns {item_id: raw_verdict}.

    Failure is contained here: a chunk that cannot be classified returns empty
    and its items fall back individually, leaving every other chunk untouched.
    """
    payload: dict[str, Any] = {
        "instructions": (
            "Answer for EVERY item. Return an object whose 'results' maps each "
            "item's 'id' to that item's verdict. Use the id exactly as given; "
            "do not reorder, merge, or omit items."
        ),
        "items": [{"id": item_key, **item_payload(item)} for item_key, item in chunk],
    }
    if shared:
        payload["shared_context"] = shared
    if output_hint is not None:
        payload["per_item_output_schema"] = output_hint

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

    for attempt in (1, 2):
        try:
            doc, _raw = await chat_json(
                llm_config=llm_config,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"},
                fallback={"results": {}},
            )
            verdicts = _results_by_id(doc, [k for k, _ in chunk])
            if verdicts:
                return verdicts
            if attempt == 1:
                logger.debug("Batch returned no usable verdicts; retrying once")
                continue
        except Exception as exc:
            if attempt == 1:
                logger.debug("Batch call failed (%s); retrying once", exc)
                continue
            logger.warning("Batch call failed twice; falling back per item: %s", exc)
    return {}


def _results_by_id(doc: Any, expected: Iterable[str]) -> dict[str, Any]:
    """Pull per-item verdicts out of the response, keyed by id.

    Accepts the mapping shape we ask for and the list-of-objects shape models
    often return anyway. **Never falls back to positional matching** — a
    reordered or partial response would silently attribute one item's verdict
    to another, which is worse than no answer at all.
    """
    if not isinstance(doc, dict):
        return {}
    results = doc.get("results")
    wanted = set(expected)

    if isinstance(results, dict):
        return {str(k): v for k, v in results.items() if str(k) in wanted}

    if isinstance(results, list):
        out: dict[str, Any] = {}
        for entry in results:
            if not isinstance(entry, dict):
                continue
            entry_id = str(entry.get("id") or entry.get("item_id") or "")
            if entry_id in wanted:
                out[entry_id] = entry
        return out

    # Some models drop the wrapper and answer with the mapping directly.
    direct = {str(k): v for k, v in doc.items() if str(k) in wanted}
    return direct
