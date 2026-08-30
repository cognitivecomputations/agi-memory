"""HMX onboarding interviews (#97): a use case fully defined in one HMX file.

An imported HMX document can carry an ``onboarding_interview`` section
(schema in ``schemas/hmx-1.7.schema.json``) -- questions the agent asks
conversationally in the first surface after import, each bound to an action
via existing machinery. This module runs that interview over the CLI
(``questionary``, per house ruling that the CLI is line-based); persistence
and binding execution live in DB functions (``db/49a_hmx_onboarding_interview.sql``)
so state survives restarts.

Not attempted here (left for follow-up work, called out in the PR): web UI
button rendering, an init-wizard-stage variant, and binding executors for
anything other than ``remember`` -- those fail loud with ``not_implemented``
rather than silently doing nothing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import asyncpg

from apps.cli_prompts import select_value, text as prompt_text

_SKIP_SENTINEL = object()


@dataclass(frozen=True)
class InterviewOutcome:
    """What happened running (or not running) an interview."""

    ran: bool
    completed: bool = False
    answered: int = 0
    skipped: int = 0
    failed_bindings: int = 0
    note: str | None = None


def _coerce_json(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


async def maybe_run_cli_interview(
    conn: asyncpg.Connection,
    document: dict[str, Any],
    *,
    interactive: bool,
) -> InterviewOutcome:
    """Run the document's onboarding_interview, if it has one.

    Returns ``ran=False`` when there is no interview section, or when
    ``interactive`` is False (e.g. ``--json`` output, or no tty) -- in the
    latter case the interview state is still started so it can be resumed
    later, and the caller is told so via ``note``.
    """
    interview = document.get("onboarding_interview")
    if not interview or not interview.get("questions"):
        return InterviewOutcome(ran=False)

    export_id = document["export_id"]
    state = _coerce_json(
        await conn.fetchval(
            "SELECT start_hmx_interview($1, $2::jsonb)",
            export_id,
            _dumps(interview),
        )
    )

    if not interactive:
        return InterviewOutcome(
            ran=False,
            note=(
                f"This import includes an onboarding interview ({len(interview['questions'])} "
                "question(s)) that was not run (non-interactive session). Its state was "
                f"recorded under export_id {export_id}."
            ),
        )

    if state["status"] == "completed":
        return InterviewOutcome(ran=False, note="This import's onboarding interview was already completed.")

    answered_ids = {a["question_id"] for a in state["answers"] if a["status"] in ("answered", "skipped")}
    answered = skipped = failed_bindings = 0

    print()
    print("This use case has a few questions for you.")
    for question in interview["questions"]:
        if question["id"] in answered_ids:
            continue
        outcome = await _ask_one(question)
        if outcome is _SKIP_SENTINEL:
            await conn.fetchval(
                "SELECT record_hmx_interview_answer($1, $2, 'skipped')",
                export_id,
                question["id"],
            )
            skipped += 1
            continue
        result = _coerce_json(
            await conn.fetchval(
                "SELECT record_hmx_interview_answer($1, $2, 'answered', $3::jsonb)",
                export_id,
                question["id"],
                _dumps(outcome),
            )
        )
        answered += 1
        binding = result.get("binding_result") or {}
        if not binding.get("success", True):
            failed_bindings += 1
            print(f"  (could not apply: {binding.get('error', 'unknown error')})")

    final_state = _coerce_json(await conn.fetchval("SELECT get_hmx_interview_state($1)", export_id))
    return InterviewOutcome(
        ran=True,
        completed=final_state["status"] == "completed",
        answered=answered,
        skipped=skipped,
        failed_bindings=failed_bindings,
    )


async def _ask_one(question: dict[str, Any]) -> Any:
    """Ask one question; returns the answer value, or _SKIP_SENTINEL."""
    skippable = bool(question.get("skippable", not question.get("required", False)))
    prompt = question["prompt"]

    if question["type"] == "choice":
        pairs = [(opt["label"], opt["value"]) for opt in question.get("options", [])]
        if skippable:
            pairs = [*pairs, ("(skip this question)", _SKIP_SENTINEL)]
        return await select_value(prompt, pairs)

    # freeform
    while True:
        answer = await prompt_text(prompt)
        answer = (answer or "").strip()
        if answer:
            return answer
        if skippable:
            return _SKIP_SENTINEL
        print("  This one needs an answer.")


def _dumps(value: Any) -> str:
    return json.dumps(value)
