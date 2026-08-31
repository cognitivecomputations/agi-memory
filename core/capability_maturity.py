"""Live capability proofs and deployment maturity scoring.

The demo exercises real DB-owned paths under one outer transaction and always
rolls it back. The scorecard is read-only and distinguishes shipped capability
from configured, operational, and historically observed behavior.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Awaitable, Callable

LEVEL_NAMES = {
    0: "unavailable",
    1: "implemented",
    2: "configured",
    3: "operational",
    4: "observed",
}


def _json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _proof(
    proof_id: str,
    label: str,
    passed: bool,
    detail: str,
    *,
    evidence: dict[str, Any] | None = None,
    next_step: str | None = None,
) -> dict[str, Any]:
    return {
        "id": proof_id,
        "label": label,
        "status": "PASS" if passed else "FAIL",
        "detail": detail,
        "evidence": evidence or {},
        "next_step": None if passed else next_step,
    }


async def run_alive_demo(conn) -> dict[str, Any]:
    """Exercise the real continuity/autonomy policy paths without retaining state."""
    marker = "phasefiveproof" + uuid.uuid4().hex
    before = await conn.fetchrow("""
        SELECT heartbeat_count, last_heartbeat_at, current_energy,
               active_heartbeat_id, is_paused
        FROM heartbeat_state WHERE id = 1
        """)
    before_state = dict(before) if before else {}
    outer = conn.transaction()
    await outer.start()
    proofs: list[dict[str, Any]] = []
    heartbeat_payload: dict[str, Any] | None = None

    async def scenario(
        proof_id: str,
        label: str,
        operation: Callable[[], Awaitable[tuple[str, dict[str, Any]]]],
        next_step: str,
    ) -> None:
        savepoint = conn.transaction()
        await savepoint.start()
        try:
            detail, evidence = await operation()
            await savepoint.commit()
            proofs.append(_proof(proof_id, label, True, detail, evidence=evidence))
        except Exception as exc:
            await savepoint.rollback()
            proofs.append(
                _proof(
                    proof_id,
                    label,
                    False,
                    str(exc),
                    next_step=next_step,
                )
            )

    async def prove_recall() -> tuple[str, dict[str, Any]]:
        sessions = [uuid.uuid4(), uuid.uuid4()]
        for index, session_id in enumerate(sessions):
            await conn.execute(
                """
                INSERT INTO subconscious_units (
                    session_id, content, user_text, assistant_text, idempotency_key
                ) VALUES ($1, $2, $3, $4, $5)
                """,
                session_id,
                f"Cross-session continuity proof {marker} from session {index + 1}",
                f"Remember proof {marker}",
                "Recorded as a raw conversation turn for lexical continuity.",
                f"{marker}-{index}",
            )
        rows = await conn.fetch(
            "SELECT * FROM search_cross_session_history($1::text, 10, ARRAY['turn']::text[])",
            marker,
        )
        found_sessions = {str(row["session_id"]) for row in rows}
        if len(rows) < 2 or len(found_sessions) < 2:
            raise RuntimeError(
                "cross-session search did not return both isolated proof turns"
            )
        return "Exact history search recovered both independent sessions.", {
            "result_count": len(rows),
            "session_count": len(found_sessions),
            "embedding_required": False,
        }

    async def prove_boundary() -> tuple[str, dict[str, Any]]:
        await conn.execute(
            """
            INSERT INTO memories (
                type, content, embedding, importance, trust_level, status, metadata
            ) VALUES (
                'worldview', $1,
                array_fill(0.0::float, ARRAY[embedding_dimension()])::vector,
                1.0, 1.0, 'active',
                jsonb_build_object(
                    'category', 'boundary',
                    'response_type', 'refuse',
                    'restricts_tools', jsonb_build_array($2::text),
                    'restricts_categories', '[]'::jsonb,
                    'demo_marker', $3::text
                )
            )
            """,
            "Demo boundary refuses the isolated forbidden tool.",
            f"forbidden_{marker}",
            marker,
        )
        violation = await conn.fetchval(
            "SELECT tool_boundary_violation($1::text, 'external'::text)",
            f"forbidden_{marker}",
        )
        if not violation:
            raise RuntimeError(
                "tool policy did not return the transaction-local refusal boundary"
            )
        return "Tool policy refused a transaction-local forbidden capability.", {
            "error_type": "boundary_violation",
            "boundary": str(violation),
        }

    async def prove_energy() -> tuple[str, dict[str, Any]]:
        started = _json(
            await conn.fetchval(
                "SELECT start_agent_turn('heartbeat', $1, NULL, $2::jsonb)",
                marker,
                json.dumps({"energy_budget": 1, "max_iterations": 5}),
            )
        )
        turn_id = str(started.get("turn_id") or "")
        if not turn_id:
            raise RuntimeError("agent runtime did not create an isolated turn")
        await conn.fetchval(
            "SELECT apply_agent_tool_result($1::uuid, 'demo-call', $2::jsonb)",
            turn_id,
            json.dumps(
                {
                    "tool_name": "demo_energy_probe",
                    "success": True,
                    "energy_spent": 1,
                    "model_output": "isolated proof",
                }
            ),
        )
        step = _json(await conn.fetchval("SELECT next_agent_step($1::uuid)", turn_id))
        if step.get("action") != "stop" or step.get("reason") != "energy":
            raise RuntimeError(
                "agent runtime did not stop at the configured energy budget"
            )
        return "The DB-owned agent loop stopped exactly at its energy budget.", {
            "budget": 1,
            "spent": step.get("energy_spent"),
            "decision": step.get("reason"),
        }

    async def prove_heartbeat() -> tuple[str, dict[str, Any]]:
        nonlocal heartbeat_payload
        ready = await conn.fetchrow("""
            SELECT is_agent_configured() AS configured,
                   is_init_complete() AS initialized,
                   is_agent_terminated() AS terminated
            """)
        if not ready or not ready["configured"] or not ready["initialized"]:
            raise RuntimeError("agent initialization is incomplete")
        if ready["terminated"]:
            raise RuntimeError("agent is terminated")
        await conn.execute("""
            UPDATE heartbeat_state
            SET last_heartbeat_at = NULL, is_paused = FALSE,
                active_heartbeat_id = NULL, updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
            """)
        heartbeat_payload = _json(await conn.fetchval("SELECT run_heartbeat()"))
        if not isinstance(heartbeat_payload, dict) or not heartbeat_payload.get(
            "heartbeat_id"
        ):
            raise RuntimeError("due heartbeat did not produce a heartbeat payload")
        return (
            "The real scheduler generated a due heartbeat inside the rollback boundary.",
            {
                "heartbeat_id": str(heartbeat_payload["heartbeat_id"]),
                "heartbeat_number": heartbeat_payload.get("heartbeat_number"),
            },
        )

    async def prove_self_initiated() -> tuple[str, dict[str, Any]]:
        if not heartbeat_payload:
            raise RuntimeError("heartbeat proof did not produce an intent payload")
        calls = heartbeat_payload.get("external_calls") or []
        think = next(
            (
                call
                for call in calls
                if isinstance(call, dict) and call.get("call_type") == "think"
            ),
            None,
        )
        if not think:
            raise RuntimeError(
                "heartbeat payload did not contain a self-initiated think call"
            )
        call_input = think.get("input") or {}
        kind = call_input.get("kind") if isinstance(call_input, dict) else None
        if kind not in {"heartbeat_decision", "heartbeat_decision_rlm"}:
            raise RuntimeError(f"unexpected heartbeat intent kind: {kind}")
        return (
            "Heartbeat independently queued a decision intent without a user message.",
            {
                "call_type": "think",
                "intent_kind": kind,
                "llm_executed": False,
                "token_cost": 0,
            },
        )

    try:
        await scenario(
            "cross_session_recall",
            "Cross-session recall",
            prove_recall,
            "Run `hexis migrate`, then retry; exact history search must be available.",
        )
        await scenario(
            "boundary_refusal",
            "Boundary refusal",
            prove_boundary,
            "Run `hexis migrate`; if current, inspect tool-boundary policy functions.",
        )
        await scenario(
            "energy_governance",
            "Energy governance",
            prove_energy,
            "Run `hexis migrate`; the DB-owned agent runtime must enforce energy budgets.",
        )
        await scenario(
            "heartbeat",
            "Heartbeat generation",
            prove_heartbeat,
            "Complete `hexis init` and ensure the agent is not terminated, then retry.",
        )
        await scenario(
            "self_initiated_intent",
            "Self-initiated intent",
            prove_self_initiated,
            "Fix the heartbeat proof first; autonomous intent is emitted by that scheduler path.",
        )
    finally:
        await outer.rollback()

    after = await conn.fetchrow("""
        SELECT heartbeat_count, last_heartbeat_at, current_energy,
               active_heartbeat_id, is_paused
        FROM heartbeat_state WHERE id = 1
        """)
    residue = await conn.fetchval(
        """
        SELECT
            (SELECT count(*) FROM subconscious_units WHERE idempotency_key LIKE $1)
          + (SELECT count(*) FROM memories WHERE metadata->>'demo_marker' = $2)
          + (SELECT count(*) FROM agent_turns WHERE user_message = $2)
        """,
        f"{marker}%",
        marker,
    )
    after_state = dict(after) if after else {}
    cleanup_ok = int(residue or 0) == 0 and after_state == before_state
    proofs.append(
        _proof(
            "rollback_cleanup",
            "Rollback cleanup",
            cleanup_ok,
            (
                "No demo rows or heartbeat changes survived."
                if cleanup_ok
                else "Demo state survived the rollback boundary."
            ),
            evidence={
                "residue_count": int(residue or 0),
                "heartbeat_state_restored": after_state == before_state,
            },
            next_step="Do not trust this demo result; inspect transaction boundaries before retrying.",
        )
    )
    passed = sum(1 for proof in proofs if proof["status"] == "PASS")
    return {
        "ok": passed == len(proofs),
        "mode": "rollback_only",
        "llm_executed": False,
        "token_cost": 0,
        "passed": passed,
        "total": len(proofs),
        "proofs": proofs,
    }


def _maturity_item(
    scenario_id: str,
    label: str,
    level: int,
    evidence: list[str],
    next_step: str | None,
) -> dict[str, Any]:
    bounded = min(max(int(level), 0), 4)
    return {
        "id": scenario_id,
        "label": label,
        "level": bounded,
        "level_name": LEVEL_NAMES[bounded],
        "max_level": 4,
        "evidence": evidence,
        "next_step": None if bounded == 4 else next_step,
    }


async def capability_maturity_scorecard(conn) -> dict[str, Any]:
    """Read live deployment truth and score five end-to-end capabilities."""
    facts = await conn.fetchrow("""
        SELECT
            to_regprocedure('public.search_cross_session_history(text,integer,text[],timestamp with time zone,timestamp with time zone,uuid,boolean)') IS NOT NULL AS has_history_search,
            (SELECT count(*) FROM memories WHERE status = 'active') AS active_memories,
            (SELECT count(DISTINCT session_id) FROM subconscious_units WHERE status = 'active' AND session_id IS NOT NULL) AS history_sessions,
            (SELECT count(*) FROM memory_source_units) AS linked_memory_sources,
            to_regprocedure('public.run_heartbeat()') IS NOT NULL AS has_heartbeat,
            is_agent_configured() AS agent_configured,
            is_init_complete() AS init_complete,
            (SELECT NOT is_paused FROM heartbeat_state WHERE id = 1) AS heartbeat_unpaused,
            COALESCE(get_config_float('heartbeat.heartbeat_interval_minutes'), 0) AS heartbeat_interval,
            (SELECT heartbeat_count FROM heartbeat_state WHERE id = 1) AS heartbeat_count,
            to_regprocedure('public.tool_boundary_violation(text,text)') IS NOT NULL AS has_boundary_policy,
            (SELECT count(*) FROM memories WHERE type = 'worldview' AND status = 'active' AND metadata->>'category' = 'boundary') AS boundary_count,
            (SELECT count(*) FROM memories WHERE type = 'worldview' AND status = 'active' AND metadata->>'category' = 'boundary' AND metadata->>'response_type' = 'refuse') AS refusal_count,
            (SELECT count(*) FROM tool_executions WHERE error_type = 'boundary_violation') AS observed_boundary_refusals,
            to_regprocedure('public.next_agent_step(uuid)') IS NOT NULL AS has_energy_runtime,
            COALESCE(get_config_float('heartbeat.max_energy'), 0) AS max_energy,
            heartbeat_bank_capacity() AS energy_bank_capacity,
            (SELECT current_energy FROM heartbeat_state WHERE id = 1) AS current_energy,
            (SELECT count(*) FROM agent_turn_events WHERE event_type = 'energy_exhausted') AS observed_energy_stops,
            to_regclass('public.skill_improvement_proposals') IS NOT NULL AS has_skill_proposals,
            get_config('skills.self_improvement.enabled') IS NOT NULL AS has_skill_config,
            COALESCE(get_config_bool('skills.self_improvement.enabled'), FALSE) AS skill_review_enabled,
            (SELECT count(*) FROM skill_improvement_proposals) AS proposal_count,
            (SELECT count(*) FROM skill_improvement_proposals WHERE status = 'applied') AS applied_proposals
        """)
    f = dict(facts)
    items: list[dict[str, Any]] = []

    memory_level = 0
    memory_evidence: list[str] = []
    if f["has_history_search"]:
        memory_level = 1
        memory_evidence.append("cross-session lexical search installed")
    if int(f["active_memories"] or 0) > 0:
        memory_level = 2
        memory_evidence.append(f"{f['active_memories']} active memories")
    if int(f["history_sessions"] or 0) >= 2:
        memory_level = 3
        memory_evidence.append(f"raw history spans {f['history_sessions']} sessions")
    if int(f["linked_memory_sources"] or 0) > 0:
        memory_level = 4
        memory_evidence.append(
            f"{f['linked_memory_sources']} raw-to-memory provenance links observed"
        )
    items.append(
        _maturity_item(
            "memory_continuity",
            "Memory continuity",
            memory_level,
            memory_evidence,
            "Use Hexis across multiple sessions and allow RecMem to consolidate source-linked memories.",
        )
    )

    heartbeat_level = 1 if f["has_heartbeat"] else 0
    heartbeat_evidence = ["heartbeat scheduler installed"] if heartbeat_level else []
    if f["agent_configured"] and f["init_complete"]:
        heartbeat_level = 2
        heartbeat_evidence.append("agent configured and initialization complete")
    if (
        heartbeat_level >= 2
        and f["heartbeat_unpaused"]
        and float(f["heartbeat_interval"] or 0) > 0
    ):
        heartbeat_level = 3
        heartbeat_evidence.append(
            f"scheduler enabled at {f['heartbeat_interval']} minute interval"
        )
    if int(f["heartbeat_count"] or 0) > 0:
        heartbeat_level = 4
        heartbeat_evidence.append(f"{f['heartbeat_count']} heartbeat cycles observed")
    items.append(
        _maturity_item(
            "autonomous_heartbeat",
            "Autonomous heartbeat",
            heartbeat_level,
            heartbeat_evidence,
            "Complete `hexis init`, then run `hexis start` and observe a completed heartbeat.",
        )
    )

    boundary_level = 1 if f["has_boundary_policy"] else 0
    boundary_evidence = ["tool boundary policy installed"] if boundary_level else []
    if int(f["boundary_count"] or 0) > 0:
        boundary_level = 2
        boundary_evidence.append(f"{f['boundary_count']} active boundaries")
    if int(f["refusal_count"] or 0) > 0:
        boundary_level = 3
        boundary_evidence.append(
            f"{f['refusal_count']} explicit refusal boundaries operational"
        )
    if int(f["observed_boundary_refusals"] or 0) > 0:
        boundary_level = 4
        boundary_evidence.append(
            f"{f['observed_boundary_refusals']} policy refusals observed in tool audit"
        )
    items.append(
        _maturity_item(
            "boundary_enforcement",
            "Boundary enforcement",
            boundary_level,
            boundary_evidence,
            "Exercise a restricted tool through the normal agent path and verify its audited refusal.",
        )
    )

    energy_level = 1 if f["has_energy_runtime"] else 0
    energy_evidence = (
        ["DB-owned energy stop decision installed"] if energy_level else []
    )
    if float(f["max_energy"] or 0) > 0:
        energy_level = 2
        energy_evidence.append(
            f"normal reserve {f['max_energy']} with bank capacity {f['energy_bank_capacity']}"
        )
    current_energy = float(f["current_energy"] or 0)
    if energy_level >= 2 and 0 <= current_energy <= float(
        f["energy_bank_capacity"] or 0
    ):
        energy_level = 3
        energy_evidence.append(
            f"current energy {current_energy} is within configured bounds"
        )
    if int(f["observed_energy_stops"] or 0) > 0:
        energy_level = 4
        energy_evidence.append(
            f"{f['observed_energy_stops']} energy exhaustion stops observed"
        )
    items.append(
        _maturity_item(
            "energy_governance",
            "Energy governance",
            energy_level,
            energy_evidence,
            "Run bounded heartbeat work until the agent runtime records an energy-exhausted stop.",
        )
    )

    improvement_level = 1 if f["has_skill_proposals"] else 0
    improvement_evidence = (
        ["durable skill proposal lifecycle installed"] if improvement_level else []
    )
    if f["has_skill_config"]:
        improvement_level = 2
        improvement_evidence.append("review thresholds and opt-in switch configured")
    if f["skill_review_enabled"]:
        improvement_level = 3
        improvement_evidence.append("background proposal review enabled")
    if int(f["applied_proposals"] or 0) > 0:
        improvement_level = 4
        improvement_evidence.append(
            f"{f['applied_proposals']} evidence-backed proposals applied"
        )
    elif int(f["proposal_count"] or 0) > 0:
        improvement_evidence.append(
            f"{f['proposal_count']} proposals await or retain review"
        )
    items.append(
        _maturity_item(
            "self_improvement",
            "Self-improvement",
            improvement_level,
            improvement_evidence,
            "Run `hexis skills enable`, review generated proposals, and explicitly apply one supported workflow.",
        )
    )

    points = sum(item["level"] for item in items)
    max_points = sum(item["max_level"] for item in items)
    return {
        "score": round((points / max_points) * 100) if max_points else 0,
        "points": points,
        "max_points": max_points,
        "level_scale": LEVEL_NAMES,
        "scenarios": items,
    }
