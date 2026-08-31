"""HMX intent policy + envelope construction (plans/hmx.md, Slice 1).

Pins the safety-critical policy matrix: which sections each export intent
carries by default, what explicit opt-in unlocks, and the envelope shape.
"""

from __future__ import annotations

import re

import pytest

from core.memory_exchange import (
    ALL_SECTIONS,
    HMX_VERSION,
    PROTECTED_SECTIONS,
    HmxPolicyError,
    HmxSchemaError,
    build_envelope,
    default_import_strategy,
    dry_run_hmx,
    export_hmx,
    import_hmx,
    iter_hmx_jsonl,
    new_export_id,
    parse_hmx_jsonl,
    resolve_export_sections,
    validate_hmx_document,
    validate_intent,
)


class TestIntentPolicy:
    def test_port_includes_everything_portable(self):
        plan = resolve_export_sections("port")
        assert set(plan.sections) == set(ALL_SECTIONS)
        assert set(plan.protected) == PROTECTED_SECTIONS

    def test_duplicate_includes_everything_portable(self):
        plan = resolve_export_sections("duplicate")
        assert set(plan.sections) == set(ALL_SECTIONS)

    def test_telepathy_excludes_protected_by_default(self):
        plan = resolve_export_sections("telepathy")
        assert set(plan.sections) == {
            "memories",
            "episodes",
            "relationships",
            "clusters",
        }
        assert plan.protected == ()
        assert "in_flight_work" not in plan.sections
        assert "audit_records" not in plan.sections

    def test_analysis_excludes_protected_by_default(self):
        plan = resolve_export_sections("analysis")
        assert plan.protected == ()

    def test_telepathy_protected_requires_explicit_opt_in(self):
        plan = resolve_export_sections(
            "telepathy", include_protected=["worldview", "narrative"]
        )
        assert "worldview" in plan.sections
        assert "narrative" in plan.sections
        assert "identity" not in plan.sections
        assert any("deliberative" in w for w in plan.warnings)

    def test_requesting_protected_without_opt_in_is_refused(self):
        with pytest.raises(HmxPolicyError, match="include_protected"):
            resolve_export_sections("analysis", sections=["memories", "worldview"])

    def test_unknown_protected_section_rejected(self):
        with pytest.raises(HmxPolicyError, match="protected sections"):
            resolve_export_sections("telepathy", include_protected=["memories"])

    def test_unknown_intent_rejected(self):
        with pytest.raises(HmxPolicyError, match="export_intent"):
            validate_intent("backup")

    def test_port_can_drop_in_flight_but_warns_on_dropping_audit(self):
        plan = resolve_export_sections(
            "port", include_in_flight_work=False, include_audit_records=False
        )
        assert "in_flight_work" not in plan.sections
        assert "audit_records" not in plan.sections
        assert any("history" in w for w in plan.warnings)

    def test_sections_are_in_canonical_order(self):
        plan = resolve_export_sections("port")
        assert plan.sections == tuple(
            s for s in ALL_SECTIONS if s in set(plan.sections)
        )

    def test_optional_sections_are_explicit_and_appended(self):
        default = resolve_export_sections("port")
        opted_in = resolve_export_sections(
            "port", include_raw_units=True, include_config=True
        )
        assert "raw_units" not in default.sections
        assert "config" not in default.sections
        assert opted_in.sections[-2:] == ("raw_units", "config")

    def test_default_strategies(self):
        assert default_import_strategy("port") == "authoritative"
        assert default_import_strategy("duplicate") == "authoritative"
        assert default_import_strategy("telepathy") == "deliberative"
        assert default_import_strategy("analysis") == "analysis_only"


class TestEnvelope:
    def _envelope(self, intent="port", **kwargs):
        plan = resolve_export_sections(intent)
        return build_envelope(
            intent=intent,
            plan=plan,
            instance_id="hexis_test",
            schema_version="0003_hmx_bootstrap_provenance",
            embedding_model="embeddinggemma:300m-qat-q4_0",
            embedding_dimension=768,
            lineage_id="11111111-2222-3333-4444-555555555555",
            relationship_edge_types=["SUPPORTS", "SUPERSEDES", "CAUSES"],
            **kwargs,
        )

    def test_envelope_required_fields(self):
        env = self._envelope()
        for field in (
            "hmx_version",
            "export_id",
            "export_intent",
            "exported_at",
            "source",
            "sections",
        ):
            assert field in env
        assert env["hmx_version"] == HMX_VERSION
        assert (
            env["source"]["hexis_lineage_id"] == "11111111-2222-3333-4444-555555555555"
        )
        assert env["source"]["embedding_dimension"] == 768

    def test_export_id_format(self):
        assert re.fullmatch(r"exp_[0-9a-f]{16}", new_export_id())
        assert re.fullmatch(r"exp_[0-9a-f]{16}", self._envelope()["export_id"])

    def test_capabilities_derive_from_given_edge_types(self):
        env = self._envelope()
        assert env["capabilities"]["relationship_edge_types"] == [
            "CAUSES",
            "SUPERSEDES",
            "SUPPORTS",
        ]
        assert "jsonl" in env["capabilities"]["formats"]
        assert "protected_section_digest_v1" in env["capabilities"]["hash_algorithms"]
        assert (
            "protected_replacement_protocol_v1"
            in env["capabilities"]["optional_features"]
        )
        assert "fast_path_verification" in env["capabilities"]["optional_features"]

    def test_export_scope_reflects_plan(self):
        env = self._envelope()
        assert set(env["export_scope"]["include_protected"]) == PROTECTED_SECTIONS
        assert env["export_scope"]["include_in_flight_work"] is True
        assert env["export_scope"]["include_audit_records"] is True

        telepathy_plan = resolve_export_sections("telepathy")
        env2 = build_envelope(
            intent="telepathy",
            plan=telepathy_plan,
            instance_id="hexis_test",
            schema_version="baseline",
            embedding_model="m",
            embedding_dimension=768,
            lineage_id="x",
            relationship_edge_types=[],
        )
        assert env2["export_scope"]["include_protected"] == []
        assert env2["export_scope"]["include_audit_records"] is False

    def test_unknown_redaction_policy_rejected(self):
        with pytest.raises(HmxPolicyError, match="redaction_policy"):
            self._envelope(redaction_policy="paranoid")

    def test_statistics_start_zeroed(self):
        env = self._envelope()
        assert env["statistics"]["total_memories"] == 0
        assert env["statistics"]["estimated_embedding_cost_units"] is None


class TestCanonicalSchema:
    def _envelope(self):
        plan = resolve_export_sections("port")
        return build_envelope(
            intent="port",
            plan=plan,
            instance_id="hexis_test",
            schema_version="0004_hmx_export_functions",
            embedding_model="embeddinggemma:300m",
            embedding_dimension=768,
            lineage_id="11111111-2222-3333-4444-555555555555",
            relationship_edge_types=["SUPPORTS"],
        )

    def test_generated_envelope_is_schema_valid(self):
        validate_hmx_document(self._envelope())

    def test_unknown_fields_and_sections_are_forward_compatible(self):
        env = self._envelope()
        env["future_header"] = {"preserve": True}
        env["sections"]["future_section"] = [{"shape": "unknown"}]
        validate_hmx_document(env)

    def test_unknown_major_version_is_rejected(self):
        env = self._envelope()
        env["hmx_version"] = "2.0"
        with pytest.raises(HmxSchemaError, match=r"\$\.hmx_version"):
            validate_hmx_document(env)

    def test_memory_requires_ref_type_and_content(self):
        env = self._envelope()
        env["sections"]["memories"] = [{"type": "semantic", "content": "x"}]
        with pytest.raises(HmxSchemaError, match="'ref' is a required property"):
            validate_hmx_document(env)

    def test_audit_union_requires_event_specific_fields(self):
        env = self._envelope()
        env["sections"]["audit_records"] = {
            "protected_section_verified_audit": [
                {
                    "audit_id": "audit-1",
                    "event_type": "protected_section_verified",
                    "event_time": "2026-07-09T12:00:00Z",
                    "sections_verified": ["worldview"],
                    "local_digest_v1": "a" * 64,
                    "imported_digest_v1": "a" * 64,
                }
            ]
        }
        with pytest.raises(HmxSchemaError, match="'source' is a required property"):
            validate_hmx_document(env)

    def test_well_formed_verified_audit_is_valid(self):
        env = self._envelope()
        env["sections"]["audit_records"] = {
            "protected_section_verified_audit": [
                {
                    "audit_id": "audit-1",
                    "event_type": "protected_section_verified",
                    "event_time": "2026-07-09T12:00:00Z",
                    "sections_verified": ["worldview"],
                    "source": {
                        "export_id": env["export_id"],
                        "origin_instance": "hexis_test",
                        "hexis_lineage_id": env["source"]["hexis_lineage_id"],
                        "export_intent": "port",
                    },
                    "local_digest_v1": "a" * 64,
                    "imported_digest_v1": "a" * 64,
                }
            ]
        }
        validate_hmx_document(env)


class TestOnboardingInterviewSchema:
    """#97: a use case fully defined in one HMX file -- the envelope's
    optional onboarding_interview section, part of the open HMX standard."""

    def _envelope(self):
        plan = resolve_export_sections("port")
        return build_envelope(
            intent="port",
            plan=plan,
            instance_id="hexis_test",
            schema_version="0004_hmx_export_functions",
            embedding_model="embeddinggemma:300m",
            embedding_dimension=768,
            lineage_id="11111111-2222-3333-4444-555555555555",
            relationship_edge_types=["SUPPORTS"],
        )

    def _question(self, **overrides):
        question = {
            "id": "q1",
            "type": "choice",
            "prompt": "What should I call you?",
            "options": [
                {"label": "Boss", "value": "boss"},
                {"label": "Friend", "value": "friend"},
            ],
            "binds": {"action": "remember", "params_template": {"content": "{{answer}}"}},
        }
        question.update(overrides)
        return question

    def test_well_formed_interview_is_valid(self):
        env = self._envelope()
        env["onboarding_interview"] = {"version": 1, "questions": [self._question()]}
        validate_hmx_document(env)

    def test_freeform_question_without_options_is_valid(self):
        env = self._envelope()
        # A freeform question legitimately omits 'options' entirely.
        question = self._question(id="q2", type="freeform", prompt="Anything else I should know?")
        del question["options"]
        env["onboarding_interview"] = {"version": 1, "questions": [question]}
        validate_hmx_document(env)

    def test_required_and_skippable_flags_are_valid(self):
        env = self._envelope()
        env["onboarding_interview"] = {
            "version": 1,
            "questions": [self._question(required=True, skippable=False)],
        }
        validate_hmx_document(env)

    def test_missing_binds_is_rejected(self):
        env = self._envelope()
        question = self._question()
        del question["binds"]
        env["onboarding_interview"] = {"version": 1, "questions": [question]}
        with pytest.raises(HmxSchemaError, match="'binds' is a required property"):
            validate_hmx_document(env)

    def test_unknown_binding_action_is_rejected(self):
        env = self._envelope()
        question = self._question(binds={"action": "launch_missiles"})
        env["onboarding_interview"] = {"version": 1, "questions": [question]}
        with pytest.raises(HmxSchemaError):
            validate_hmx_document(env)

    def test_option_missing_value_is_rejected(self):
        env = self._envelope()
        question = self._question(options=[{"label": "Boss"}])
        env["onboarding_interview"] = {"version": 1, "questions": [question]}
        with pytest.raises(HmxSchemaError, match="'value' is a required property"):
            validate_hmx_document(env)

    def test_document_without_interview_section_still_validates(self):
        # The section is optional -- every existing HMX document stays valid.
        validate_hmx_document(self._envelope())


class TestJsonlTransport:
    def _envelope(self):
        plan = resolve_export_sections(
            "analysis", include_raw_units=True, include_config=True
        )
        env = build_envelope(
            intent="analysis",
            plan=plan,
            instance_id="hexis_test",
            schema_version="0008_hmx_protected_import",
            embedding_model="embeddinggemma:300m",
            embedding_dimension=768,
            lineage_id="11111111-2222-3333-4444-555555555555",
            relationship_edge_types=["SUPPORTS"],
        )
        env["sections"] = {
            "memories": [],
            "relationships": [],
            "raw_units": [],
            "config": {"agent.name": "test"},
            "future_section": {"preserve": True},
        }
        return env

    def test_jsonl_round_trip_preserves_empty_config_and_future_sections(self):
        env = self._envelope()
        restored = parse_hmx_jsonl(iter_hmx_jsonl(env))
        assert restored == env
        validate_hmx_document(restored)

    def test_jsonl_requires_envelope_first(self):
        with pytest.raises(HmxSchemaError, match="first record must be envelope"):
            parse_hmx_jsonl(['{"record_type":"memory","data":{}}'])

    def test_jsonl_reports_malformed_line(self):
        with pytest.raises(HmxSchemaError, match="line 2"):
            parse_hmx_jsonl(
                [
                    '{"record_type":"envelope","data":{}}',
                    "not-json",
                ]
            )


class TestImportPreflight:
    def _envelope(self):
        plan = resolve_export_sections("telepathy")
        return build_envelope(
            intent="telepathy",
            plan=plan,
            instance_id="hexis_source",
            schema_version="0007_hmx_additive_import",
            embedding_model="embeddinggemma:300m",
            embedding_dimension=768,
            lineage_id="11111111-2222-3333-4444-555555555555",
            relationship_edge_types=[],
        )

    @pytest.mark.asyncio
    async def test_unknown_major_rejected_before_database_access(self):
        env = self._envelope()
        env["hmx_version"] = "2.0"
        with pytest.raises(HmxSchemaError, match="unsupported HMX major"):
            await import_hmx(None, env)

    @pytest.mark.asyncio
    async def test_authoritative_strategy_fails_explicitly(self):
        env = self._envelope()
        with pytest.raises(HmxPolicyError, match="authoritative replacement"):
            await import_hmx(None, env, strategy="authoritative")

    @pytest.mark.asyncio
    async def test_malformed_sections_fail_before_database_access(self):
        env = self._envelope()
        env["sections"] = []
        with pytest.raises(HmxSchemaError, match=r"\$\.sections"):
            await dry_run_hmx(None, env)

    @pytest.mark.asyncio
    async def test_strict_redaction_rejects_raw_units_before_database_access(self):
        with pytest.raises(HmxPolicyError, match="strict redaction excludes raw"):
            await export_hmx(
                None,
                intent="analysis",
                include_raw_units=True,
                redaction_policy="strict",
            )
