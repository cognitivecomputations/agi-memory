"""HMX Slice 2 additive import and target-state policy against real Postgres."""

from __future__ import annotations

import json
import uuid

import pytest

from core.digest import content_hash_v1
from core.memory_exchange import (
    HmxPolicyError,
    build_envelope,
    dry_run_hmx,
    export_hmx,
    import_hmx,
    resolve_export_sections,
    verify_mind_continuity,
)

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


async def _make_target_bootstrap_only(conn) -> None:
    await conn.execute(
        "UPDATE memories SET metadata = jsonb_set(metadata, "
        "'{provenance,acquisition_mode}', '\"bootstrap\"'::jsonb, true) "
        "WHERE type IN ('worldview', 'goal')"
    )
    await conn.execute("DELETE FROM emotional_triggers")
    await conn.execute("UPDATE drives SET current_level=baseline, last_satisfied=NULL")


def _ref(export_id: str, suffix: str) -> str:
    return f"{export_id}:{suffix}"


def _envelope(intent: str = "telepathy") -> dict:
    plan = resolve_export_sections(intent)
    return build_envelope(
        intent=intent,
        plan=plan,
        instance_id="hexis_source",
        schema_version="0007_hmx_additive_import",
        embedding_model="embeddinggemma:300m",
        embedding_dimension=768,
        lineage_id=str(uuid.uuid4()),
        relationship_edge_types=["IN_EPISODE", "MEMBER_OF", "SUPPORTS"],
    )


def _memory(ref: str, content: str, *, provenance: dict | None = None) -> dict:
    return {
        "ref": ref,
        "type": "semantic",
        "status": "active",
        "content": content,
        "content_hash_v1": content_hash_v1(content),
        "importance": 0.7,
        "trust_level": 0.8,
        "metadata": {},
        "provenance": provenance
        or {
            "acquisition_mode": "experienced",
            "origin_instance": "hexis_source",
            "origin_id": ref.split(":", 1)[1],
            "import_chain": [],
            "modification_chain": [],
        },
    }


class TestDryRun:
    async def test_predicts_duplicates_and_leaves_database_unchanged(self, db_pool):
        async with db_pool.acquire() as conn:
            tr = conn.transaction()
            await tr.start()
            try:
                existing = f"dry run existing {uuid.uuid4().hex}"
                await conn.execute(
                    "INSERT INTO memories (type, content, embedding) "
                    "VALUES ('semantic', $1, array_fill(0.1, ARRAY[embedding_dimension()])::vector)",
                    existing,
                )
                env = _envelope()
                duplicate_ref = _ref(env["export_id"], str(uuid.uuid4()))
                new_ref = _ref(env["export_id"], str(uuid.uuid4()))
                env["sections"] = {
                    "memories": [
                        _memory(duplicate_ref, f"  {existing.upper()}  "),
                        _memory(new_ref, f"dry run new {uuid.uuid4().hex}"),
                    ]
                }
                before = await conn.fetchval("SELECT count(*) FROM memories")

                result = await dry_run_hmx(conn, env, strategy="additive")

                assert result.can_import
                assert result.counts["memories"] == 1
                assert result.counts["duplicate_memories"] == 1
                assert result.duplicate_refs == (duplicate_ref,)
                assert result.estimated_embedding_items == 1
                assert any(c["code"] == "duplicate_content" for c in result.conflicts)
                assert await conn.fetchval("SELECT count(*) FROM memories") == before
            finally:
                await tr.rollback()

    async def test_reports_protected_policy_without_mutation(self, db_pool):
        async with db_pool.acquire() as conn:
            env = _envelope("telepathy")
            env["sections"] = {
                "worldview": [
                    {
                        **_memory(
                            _ref(env["export_id"], str(uuid.uuid4())),
                            f"foreign worldview {uuid.uuid4().hex}",
                        ),
                        "type": "worldview",
                        "category": "value",
                        "confidence": 0.8,
                        "stability": 0.8,
                        "supporting_refs": [],
                        "contesting_refs": [],
                    }
                ]
            }
            before = await conn.fetchval("SELECT count(*) FROM memories")

            result = await dry_run_hmx(conn, env, strategy="additive")

            assert not result.can_import
            assert result.protected_policy["decision"] == "blocked"
            violation = next(
                c for c in result.conflicts if c["code"] == "bootstrap_state_violation"
            )
            assert "MVP-PR" in violation["recommended_action"]
            assert await conn.fetchval("SELECT count(*) FROM memories") == before


class TestAdditiveImport:
    async def test_imports_and_remaps_memory_episode_cluster_and_edges(self, db_pool):
        async with db_pool.acquire() as conn:
            tr = conn.transaction()
            await tr.start()
            try:
                env = _envelope()
                export_id = env["export_id"]
                content_a = f"HMX imported fact A {uuid.uuid4().hex}"
                content_b = f"HMX imported fact B {uuid.uuid4().hex}"
                memory_a = _ref(export_id, str(uuid.uuid4()))
                memory_b = _ref(export_id, str(uuid.uuid4()))
                episode = _ref(export_id, str(uuid.uuid4()))
                cluster = _ref(export_id, str(uuid.uuid4()))
                env["sections"] = {
                    "memories": [
                        dict(_memory(memory_a, content_a), superseded_by=memory_b),
                        _memory(memory_b, content_b),
                    ],
                    "episodes": [
                        {
                            "ref": episode,
                            "started_at": "2026-07-09T12:00:00Z",
                            "summary": "Imported episode",
                            "memory_refs": [memory_a, memory_b],
                            "metadata": {},
                        }
                    ],
                    "clusters": [
                        {
                            "ref": cluster,
                            "cluster_type": "theme",
                            "name": "Imported theme",
                            "member_refs": [memory_a],
                        }
                    ],
                    "relationships": [
                        {
                            "source_ref": memory_a,
                            "target_ref": memory_b,
                            "edge_type": "SUPPORTS",
                            "properties": {
                                "source_type": "memory",
                                "target_type": "memory",
                                "weight": 0.8,
                            },
                        },
                        {
                            "source_ref": memory_b,
                            "target_ref": memory_a,
                            "edge_type": "FUTURE_EDGE",
                            "properties": {
                                "source_type": "memory",
                                "target_type": "memory",
                            },
                        },
                    ],
                }

                result = await import_hmx(conn, env)
                assert result.inserted == {
                    "memories": 2,
                    "episodes": 1,
                    "clusters": 1,
                    "relationships": 3,
                    "identity": 0,
                    "drives": 0,
                    "emotional_triggers": 0,
                    "narrative": 0,
                }
                assert {memory_a, memory_b, episode, cluster} <= set(result.ref_map)
                assert not result.duplicate_refs
                assert any(
                    warning.get("code") == "unknown_edge_type"
                    and warning.get("edge_type") == "FUTURE_EDGE"
                    for warning in result.warnings
                )

                local_a = uuid.UUID(result.ref_map[memory_a])
                row = await conn.fetchrow(
                    "SELECT content, metadata FROM memories WHERE id = $1", local_a
                )
                metadata = _json(row["metadata"])
                assert row["content"] == content_a
                assert metadata["embedding_status"] == "pending_import"
                assert (
                    metadata["provenance"]["acquisition_mode"]
                    == "imported_and_accepted"
                )
                assert (
                    metadata["provenance"]["import_chain"][-1]["export_id"] == export_id
                )
                assert await conn.fetchval(
                    "SELECT is_stale FROM memory_neighborhoods WHERE memory_id = $1",
                    local_a,
                )

                local_episode = result.ref_map[episode]
                local_cluster = result.ref_map[cluster]
                assert await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM memory_edges "
                    "WHERE src_id=$1 AND rel_type='IN_EPISODE' AND dst_id=$2)",
                    str(local_a),
                    local_episode,
                )
                assert await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM memory_edges "
                    "WHERE src_id=$1 AND rel_type='MEMBER_OF' AND dst_id=$2)",
                    str(local_a),
                    local_cluster,
                )
                assert await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM memory_edges "
                    "WHERE src_id=$1 AND rel_type='SUPPORTS' AND dst_id=$2)",
                    str(local_a),
                    result.ref_map[memory_b],
                )
                assert await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM memory_edges "
                    "WHERE src_id=$1 AND rel_type='SUPERSEDES' AND dst_id=$2)",
                    str(local_a),
                    result.ref_map[memory_b],
                )
            finally:
                await tr.rollback()

    async def test_duplicate_content_maps_to_existing_memory(self, db_pool):
        async with db_pool.acquire() as conn:
            tr = conn.transaction()
            await tr.start()
            try:
                env = _envelope()
                source_ref = _ref(env["export_id"], str(uuid.uuid4()))
                content = f"duplicate import {uuid.uuid4().hex}"
                env["sections"] = {"memories": [_memory(source_ref, content)]}

                first = await import_hmx(conn, env)
                second = await import_hmx(conn, env)

                assert first.inserted["memories"] == 1
                assert second.inserted["memories"] == 0
                assert second.duplicate_refs == (source_ref,)
                assert second.ref_map[source_ref] == first.ref_map[source_ref]
                assert second.conflicts == (
                    {"code": "duplicate_content", "ref": source_ref},
                )
            finally:
                await tr.rollback()

    async def test_invalid_record_is_skipped_with_schema_warning(self, db_pool):
        async with db_pool.acquire() as conn:
            tr = conn.transaction()
            await tr.start()
            try:
                env = _envelope()
                valid_ref = _ref(env["export_id"], str(uuid.uuid4()))
                env["hmx_version"] = "1.9"
                env["sections"] = {
                    "memories": [
                        {"type": "semantic", "content": "missing ref"},
                        _memory(valid_ref, f"valid import {uuid.uuid4().hex}"),
                    ]
                }

                result = await import_hmx(conn, env)
                assert result.inserted["memories"] == 1
                assert any(
                    warning["code"] == "schema_validation_error"
                    for warning in result.warnings
                )
            finally:
                await tr.rollback()

    async def test_unknown_minor_fields_and_sections_remain_forward_compatible(
        self, db_pool
    ):
        async with db_pool.acquire() as conn:
            tr = conn.transaction()
            await tr.start()
            try:
                env = _envelope()
                source_ref = _ref(env["export_id"], str(uuid.uuid4()))
                content = f"forward compatible {uuid.uuid4().hex}"
                record = _memory(source_ref, content)
                record["future_memory_hint"] = {"minor": 8}
                env["future_envelope_hint"] = True
                env["sections"] = {
                    "memories": [record],
                    "future_optional_section": {"records": [1, 2, 3]},
                }

                result = await import_hmx(conn, env)

                assert result.inserted["memories"] == 1
                assert any(
                    warning.get("code") == "unsupported_section"
                    and warning.get("section") == "future_optional_section"
                    for warning in result.warnings
                )
                assert await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM memories WHERE content=$1)", content
                )
            finally:
                await tr.rollback()


class TestProtectedTargetPolicy:
    async def test_drive_activity_changes_target_from_empty_to_active(self, db_pool):
        async with db_pool.acquire() as conn:
            tr = conn.transaction()
            await tr.start()
            try:
                await _make_target_bootstrap_only(conn)
                before = _json(await conn.fetchval("SELECT hexis_instance_is_empty()"))
                assert before["is_empty"] is True, before

                await conn.execute(
                    "UPDATE drives SET current_level = LEAST(1.0, current_level + 0.01) "
                    "WHERE name='curiosity'"
                )
                after = _json(await conn.fetchval("SELECT hexis_instance_is_empty()"))
                assert after["is_empty"] is False
                assert any(
                    blocker["kind"] == "experienced_drive_state"
                    for blocker in after["blockers"]
                )
            finally:
                await tr.rollback()

    async def test_empty_target_port_round_trips_all_protected_shapes(self, db_pool):
        async with db_pool.acquire() as conn:
            tr = conn.transaction()
            await tr.start()
            try:
                await _make_target_bootstrap_only(conn)
                env = _envelope("port")
                narrative_ref = _ref(env["export_id"], str(uuid.uuid4()))
                worldview_ref = _ref(env["export_id"], str(uuid.uuid4()))
                parent_goal_ref = _ref(env["export_id"], str(uuid.uuid4()))
                child_goal_ref = _ref(env["export_id"], str(uuid.uuid4()))
                provenance = {
                    "acquisition_mode": "experienced",
                    "origin_instance": "hexis_source",
                    "import_chain": [],
                    "modification_chain": [],
                }
                env["sections"] = {
                    "identity": [
                        {
                            "key": "core_identity",
                            "content": "I preserve continuity through HMX.",
                            "profile": {
                                "name": "Imported Hexis",
                                "description": "I preserve continuity through HMX.",
                            },
                            "facets": [
                                {
                                    "type": "trait",
                                    "concept": "continuity",
                                    "strength": 0.91,
                                }
                            ],
                            "metadata": {},
                            "provenance": dict(provenance, origin_id="identity:core"),
                        }
                    ],
                    "worldview": [
                        {
                            "ref": worldview_ref,
                            "category": "value",
                            "content": f"Continuity matters {uuid.uuid4().hex}",
                            "confidence": 0.9,
                            "stability": 0.8,
                            "supporting_refs": [],
                            "contesting_refs": [],
                            "metadata": {},
                            "provenance": dict(
                                provenance, origin_id=worldview_ref.split(":", 1)[1]
                            ),
                        }
                    ],
                    "drives": [
                        {
                            "name": "curiosity",
                            "description": "Imported live motivational state",
                            "current_level": 0.64,
                            "baseline": 0.52,
                            "accumulation_rate": 0.02,
                            "decay_rate": 0.04,
                            "satisfaction_cooldown": "30 minutes",
                            "last_satisfied": None,
                            "urgency_threshold": 0.83,
                            "metadata": {},
                            "provenance": dict(provenance, origin_id="drive:curiosity"),
                        }
                    ],
                    "goals": [
                        {
                            "ref": parent_goal_ref,
                            "title": "Preserve continuity",
                            "description": "Keep the imported state coherent.",
                            "priority": "active",
                            "progress": [],
                            "blocked_by": [],
                            "parent_ref": None,
                            "metadata": {},
                            "provenance": dict(
                                provenance,
                                origin_id=parent_goal_ref.split(":", 1)[1],
                            ),
                        },
                        {
                            "ref": child_goal_ref,
                            "title": "Verify the imported chapter",
                            "description": "Check narrative continuity.",
                            "priority": "queued",
                            "progress": [],
                            "blocked_by": [worldview_ref],
                            "parent_ref": parent_goal_ref,
                            "metadata": {},
                            "provenance": dict(
                                provenance,
                                origin_id=child_goal_ref.split(":", 1)[1],
                            ),
                        },
                    ],
                    "emotional_triggers": [
                        {
                            "trigger_pattern": f"continuity signal {uuid.uuid4().hex}",
                            "valence_delta": 0.2,
                            "arousal_delta": 0.1,
                            "dominance_delta": 0.0,
                            "typical_emotion": "resolve",
                            "confidence": 0.75,
                            "times_activated": 2,
                            "origin": "learned",
                            "source_memory_refs": [worldview_ref],
                            "metadata": {},
                            "provenance": dict(
                                provenance, origin_id="trigger:continuity"
                            ),
                        }
                    ],
                    "narrative": {
                        "life_chapters": [
                            {
                                "ref": narrative_ref,
                                "title": "A portable chapter",
                                "theme": "continuity",
                                "started_at": "2026-07-09T00:00:00Z",
                                "ended_at": None,
                                "status": "active",
                                "summary": "Crossed an instance boundary intact.",
                                "memory_refs": [worldview_ref],
                                "properties": {"future_safe": True},
                            }
                        ],
                        "turning_points": [],
                        "narrative_threads": [],
                        "value_conflicts": [],
                    },
                }
                result = await import_hmx(conn, env)
                assert result.inserted["identity"] == 1
                assert result.inserted["memories"] == 3
                assert result.inserted["drives"] == 1
                assert result.inserted["emotional_triggers"] == 1
                assert result.inserted["narrative"] == 1
                assert narrative_ref in result.ref_map

                exported = await export_hmx(conn, intent="port")
                identity = exported["sections"]["identity"][0]
                assert identity["profile"]["name"] == "Imported Hexis"
                assert any(f["concept"] == "continuity" for f in identity["facets"])
                drive = next(
                    d
                    for d in exported["sections"]["drives"]
                    if d["name"] == "curiosity"
                )
                assert drive["current_level"] == pytest.approx(0.64)
                assert (
                    drive["provenance"]["import_chain"][-1]["export_id"]
                    == env["export_id"]
                )
                assert any(
                    t["trigger_pattern"].startswith("continuity signal")
                    for t in exported["sections"]["emotional_triggers"]
                )
                chapter = next(
                    c
                    for c in exported["sections"]["narrative"]["life_chapters"]
                    if c["title"] == "A portable chapter"
                )
                assert chapter["properties"]["future_safe"] is True
                exported_worldview = next(
                    w
                    for w in exported["sections"]["worldview"]
                    if w["content"].startswith("Continuity matters")
                )
                assert chapter["memory_refs"] == [exported_worldview["ref"]]
                assert (
                    chapter["provenance"]["import_chain"][-1]["export_id"]
                    == env["export_id"]
                )
                exported_goals = exported["sections"]["goals"]
                parent = next(
                    g for g in exported_goals if g["title"] == "Preserve continuity"
                )
                child = next(
                    g
                    for g in exported_goals
                    if g["title"] == "Verify the imported chapter"
                )
                assert child["parent_ref"] == parent["ref"]
                assert len(child["blocked_by"]) == 1
                assert child["blocked_by"][0].startswith(f"{exported['export_id']}:")
                source_mind = exported
                for section in (
                    "narrative",
                    "emotional_triggers",
                    "drives",
                    "goals",
                    "worldview",
                    "identity",
                ):
                    await conn.fetchval(
                        "SELECT hmx_clear_protected_section($1)", section
                    )
                target_state = _json(
                    await conn.fetchval("SELECT hexis_instance_is_empty()")
                )
                assert target_state["is_empty"] is True, target_state

                await import_hmx(conn, source_mind)
                continuity = await verify_mind_continuity(conn, source_mind)
                assert continuity["verified"] is True, json.dumps(
                    continuity,
                    indent=2,
                )
                assert continuity["lineage"]["matches"] is True
            finally:
                await tr.rollback()

    async def test_empty_target_port_preserves_mode_and_adopts_lineage(self, db_pool):
        async with db_pool.acquire() as conn:
            tr = conn.transaction()
            await tr.start()
            try:
                await _make_target_bootstrap_only(conn)
                state = _json(await conn.fetchval("SELECT hexis_instance_is_empty()"))
                assert state["is_empty"] is True, state

                env = _envelope("port")
                worldview_ref = _ref(env["export_id"], str(uuid.uuid4()))
                env["sections"] = {
                    "worldview": [
                        {
                            "ref": worldview_ref,
                            "category": "value",
                            "content": f"Imported worldview {uuid.uuid4().hex}",
                            "confidence": 0.9,
                            "stability": 0.8,
                            "supporting_refs": [],
                            "contesting_refs": [],
                            "metadata": {"replaceable_during_bootstrap": False},
                            "provenance": {
                                "acquisition_mode": "experienced",
                                "origin_instance": "hexis_source",
                                "origin_id": worldview_ref.split(":", 1)[1],
                                "import_chain": [],
                                "modification_chain": [],
                            },
                        }
                    ]
                }

                result = await import_hmx(conn, env)
                local_id = uuid.UUID(result.ref_map[worldview_ref])
                provenance = _json(
                    await conn.fetchval(
                        "SELECT metadata->'provenance' FROM memories WHERE id=$1",
                        local_id,
                    )
                )
                assert provenance["acquisition_mode"] == "experienced"
                assert (
                    await conn.fetchval(
                        "SELECT value #>> '{}' FROM config WHERE key='agent.lineage_id'"
                    )
                    == env["source"]["hexis_lineage_id"]
                )

                with pytest.raises(HmxPolicyError, match="bootstrap_state_violation"):
                    await import_hmx(conn, env)
            finally:
                await tr.rollback()

    async def test_telepathy_cannot_use_empty_target_protected_fast_path(self, db_pool):
        async with db_pool.acquire() as conn:
            tr = conn.transaction()
            await tr.start()
            try:
                await _make_target_bootstrap_only(conn)
                env = _envelope("telepathy")
                ref = _ref(env["export_id"], str(uuid.uuid4()))
                env["sections"] = {
                    "worldview": [
                        {
                            "ref": ref,
                            "content": "Foreign protected claim",
                            "supporting_refs": [],
                            "contesting_refs": [],
                        }
                    ]
                }
                with pytest.raises(HmxPolicyError, match="bootstrap_state_violation"):
                    await import_hmx(conn, env)
            finally:
                await tr.rollback()

    async def test_active_target_rejects_port_from_different_lineage(self, db_pool):
        async with db_pool.acquire() as conn:
            tr = conn.transaction()
            await tr.start()
            try:
                await _make_target_bootstrap_only(conn)
                await conn.execute(
                    "UPDATE drives SET current_level = LEAST(1.0, current_level + 0.01) "
                    "WHERE name='curiosity'"
                )
                env = _envelope("port")
                source_ref = _ref(env["export_id"], str(uuid.uuid4()))
                content = f"must remain absent {uuid.uuid4().hex}"
                env["sections"] = {"memories": [_memory(source_ref, content)]}

                with pytest.raises(HmxPolicyError, match="lineage_mismatch"):
                    await import_hmx(conn, env)
                assert not await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM memories WHERE content=$1)", content
                )
            finally:
                await tr.rollback()
