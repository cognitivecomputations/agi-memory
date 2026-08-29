"""Tests for Skills Marketplace features (J.1-J.4)."""

import json
import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from skills.base import (
    InstallMethod,
    SkillCategory,
    SkillContext,
    SkillSpec,
)
from skills.loader import (
    discover_skill_dirs,
    install_skill_deps,
    load_skills,
    load_skills_from_dir,
)
from tests.utils import get_test_identifier

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


# ============================================================================
# J.1: SkillSpec Extended Fields
# ============================================================================


class TestSkillSpecExtended:
    """Test the extended SkillSpec fields (category, bins, env, install, etc.)."""

    def test_category_field(self):
        spec = SkillSpec(
            name="test",
            description="test",
            content="test",
            category=SkillCategory.RESEARCH,
        )
        assert spec.category == SkillCategory.RESEARCH

    def test_default_category(self):
        spec = SkillSpec(name="test", description="test", content="test")
        assert spec.category == SkillCategory.OTHER

    def test_requires_bins(self):
        spec = SkillSpec(
            name="test",
            description="test",
            content="test",
            requires_bins=["git", "curl"],
        )
        assert spec.requires_bins == ["git", "curl"]

    def test_requires_env(self):
        spec = SkillSpec(
            name="test",
            description="test",
            content="test",
            requires_env=["OPENAI_API_KEY"],
        )
        assert spec.requires_env == ["OPENAI_API_KEY"]

    def test_os_support(self):
        spec = SkillSpec(
            name="test",
            description="test",
            content="test",
            os_support=["darwin", "linux"],
        )
        assert "darwin" in spec.os_support

    def test_bound_tools(self):
        spec = SkillSpec(
            name="test",
            description="test",
            content="test",
            bound_tools=["web_search", "recall"],
        )
        assert spec.bound_tools == ["web_search", "recall"]

    def test_enabled_flag(self):
        spec = SkillSpec(name="test", description="test", content="test")
        assert spec.enabled is True

        spec2 = SkillSpec(
            name="test", description="test", content="test", enabled=False
        )
        assert spec2.enabled is False

    def test_install_methods(self):
        method = InstallMethod(kind="brew", package="ripgrep", bins=["rg"])
        assert method.kind == "brew"
        assert method.package == "ripgrep"
        assert method.bins == ["rg"]

    def test_install_method_from_dict(self):
        m = InstallMethod.from_dict(
            {
                "kind": "pip",
                "package": "requests",
                "bins": [],
            }
        )
        assert m.kind == "pip"
        assert m.package == "requests"

    def test_install_method_from_dict_formula(self):
        """Formula key should map to package."""
        m = InstallMethod.from_dict({"kind": "brew", "formula": "jq", "bins": ["jq"]})
        assert m.package == "jq"

    def test_from_frontmatter_with_extended_fields(self):
        metadata = {
            "name": "twitter-research",
            "description": "Twitter research skill",
            "category": "research",
            "requires": {
                "tools": ["search_twitter"],
                "config": [],
                "bins": ["curl"],
                "env": ["TWITTER_BEARER_TOKEN"],
            },
            "os_support": ["darwin", "linux"],
            "bound_tools": ["search_twitter"],
            "install": [
                {"kind": "brew", "package": "curl", "bins": ["curl"]},
            ],
            "contexts": ["heartbeat", "chat"],
        }
        content = "# Twitter Research\nUse search_twitter tool."

        spec = SkillSpec.from_frontmatter(metadata, content)
        assert spec.name == "twitter-research"
        assert spec.category == SkillCategory.RESEARCH
        assert spec.requires_bins == ["curl"]
        assert spec.requires_env == ["TWITTER_BEARER_TOKEN"]
        assert spec.bound_tools == ["search_twitter"]
        assert len(spec.install_methods) == 1
        assert spec.install_methods[0].kind == "brew"
        assert "darwin" in spec.os_support

    def test_from_frontmatter_invalid_category(self):
        metadata = {"name": "test", "description": "test", "category": "nonexistent"}
        spec = SkillSpec.from_frontmatter(metadata, "content")
        assert spec.category == SkillCategory.OTHER

    def test_from_frontmatter_string_os_support(self):
        metadata = {"name": "test", "description": "test", "os_support": "darwin"}
        spec = SkillSpec.from_frontmatter(metadata, "content")
        assert spec.os_support == ["darwin"]

    def test_from_frontmatter_string_bound_tools(self):
        metadata = {"name": "test", "description": "test", "bound_tools": "recall"}
        spec = SkillSpec.from_frontmatter(metadata, "content")
        assert spec.bound_tools == ["recall"]


# ============================================================================
# Skill-first runtime
# ============================================================================


class TestSkillRuntimeSelection:
    async def test_default_chat_exposes_the_read_only_floor_but_nothing_that_acts(
        self, db_pool, monkeypatch
    ):
        from core.tools import ToolContext, create_default_registry
        from services.skill_runtime import select_skills

        async def flat_semantic_scores(_pool, skills, _query):
            # CI's hash-derived fake vectors deliberately carry no semantics.
            # A flat distribution is the production selector's representation
            # of an ordinary continuity query with no specialized outlier.
            return {skill.name: 0.5 for skill in skills}

        monkeypatch.setattr(
            "services.skill_runtime._semantic_scores", flat_semantic_scores
        )

        registry = create_default_registry(db_pool)
        selection = await select_skills(
            registry,
            ToolContext.CHAT,
            query="what did we decide last time",
        )

        assert [s.name for s in selection.skills] == ["core-memory"], (
            selection.considered[:10]
        )
        assert {
            "list_skills",
            "use_skill",
            "recall",
            "remember",
        } <= selection.allowed_tool_names
        # A live turn always gets the read-only everyday floor: a selector that
        # guesses the topic wrongly must still leave the agent able to look
        # things up. Measured before this existed: seven of ten ordinary
        # requests reached core-memory alone and could not search the web.
        assert {
            "web_search",
            "calendar_events",
            "search_contacts",
        } <= selection.allowed_tool_names
        # Tools that *act* stay gated — the floor is for answering, not doing.
        assert "shell" not in selection.allowed_tool_names
        assert "email_send" not in selection.allowed_tool_names
        assert "write_file" not in selection.allowed_tool_names

    async def test_research_query_activates_research_without_unrelated_integrations(
        self, db_pool
    ):
        from core.tools import ToolContext, create_default_registry
        from services.skill_runtime import select_skills

        registry = create_default_registry(db_pool)
        selection = await select_skills(
            registry,
            ToolContext.CHAT,
            query="research current postgres age docs",
        )
        names = [s.name for s in selection.skills]

        # The claim in this test's name: research activates, unrelated
        # integrations do not. Not an exact roster — under semantic selection a
        # genuinely related skill activating is the feature working. ("postgres
        # age docs" also reaching self-inspection, which inspects the live
        # Postgres schema, is a reasonable read of the request.)
        assert "research" in names
        assert {"web_search", "web_fetch"} <= selection.allowed_tool_names
        assert "twitter_search" not in selection.allowed_tool_names
        assert "youtube_channel_stats" not in selection.allowed_tool_names
        assert "twitter-research" not in names
        assert "youtube-analytics" not in names

    async def test_default_heartbeat_keeps_email_digest_gated_by_autonomy_config(
        self, db_pool
    ):
        from core.tools import ToolContext, create_default_registry
        from services.skill_runtime import select_skills

        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM config WHERE key = 'integrations.gmail.heartbeat_digest_enabled'"
            )

        registry = create_default_registry(db_pool)
        selection = await select_skills(
            registry,
            ToolContext.HEARTBEAT,
            query="run heartbeat cycle",
        )
        names = [s.name for s in selection.skills]

        assert "email-digest" not in names
        assert not {
            "gmail_setup_status",
            "email_list",
            "email_read",
            "email_search",
        } & (selection.allowed_tool_names)

    async def test_heartbeat_exposes_email_digest_after_explicit_autonomous_gmail_authorization(
        self, db_pool
    ):
        from core.tools import ToolContext, create_default_registry
        from services.skill_runtime import select_skills

        marker = get_test_identifier("heartbeat-gmail-digest")
        account = f"{marker}@example.com"

        try:
            async with db_pool.acquire() as conn:
                raw_attempt = await conn.fetchval(
                    """
                    SELECT start_connection_attempt(
                        'gmail',
                        '["read", "search"]'::jsonb,
                        ARRAY[]::text[],
                        '{}'::jsonb,
                        NULL,
                        NULL,
                        'test',
                        $1,
                        CURRENT_TIMESTAMP + INTERVAL '10 minutes'
                    )
                    """,
                    marker,
                )
                attempt = (
                    json.loads(raw_attempt)
                    if isinstance(raw_attempt, str)
                    else raw_attempt
                )
                await conn.fetchval(
                    """
                    SELECT complete_connection_attempt(
                        $1::uuid,
                        $2,
                        $2,
                        'integration.gmail.default',
                        $3::text[],
                        '["read", "search"]'::jsonb,
                        '{"test": true}'::jsonb
                    )
                    """,
                    attempt["attempt_id"],
                    account,
                    [
                        "https://www.googleapis.com/auth/userinfo.email",
                        "https://www.googleapis.com/auth/gmail.readonly",
                    ],
                )
                await conn.execute(
                    "SELECT set_config('integrations.gmail.heartbeat_digest_enabled', 'true'::jsonb)"
                )

            registry = create_default_registry(db_pool)
            selection = await select_skills(
                registry,
                ToolContext.HEARTBEAT,
                query="run heartbeat cycle",
            )
            names = [s.name for s in selection.skills]

            assert "email-digest" in names
            assert {
                "gmail_setup_status",
                "email_list",
                "email_read",
                "email_search",
            } <= (selection.allowed_tool_names)
        finally:
            async with db_pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM integration_connections WHERE connector_id = 'gmail' AND account_key = $1",
                    account,
                )
                await conn.execute(
                    "DELETE FROM connection_attempts WHERE source_session_id = $1",
                    marker,
                )
                await conn.execute(
                    "DELETE FROM config WHERE key = 'integrations.gmail.heartbeat_digest_enabled'"
                )

    async def test_explicit_skill_request_activates_skill_authoring(self, db_pool):
        from core.tools import ToolContext, create_default_registry
        from services.skill_runtime import select_skills

        registry = create_default_registry(db_pool)
        selection = await select_skills(
            registry,
            ToolContext.CHAT,
            query="create a reusable skill for weekly reviews",
        )
        names = [s.name for s in selection.skills]

        assert "skill-authoring" in names
        assert "author_skill" in selection.allowed_tool_names
        assert "propose_skill" in selection.allowed_tool_names

    async def test_selection_carries_full_catalog_for_prompt_index(self, db_pool):
        from core.tools import ToolContext, create_default_registry
        from services.skill_runtime import select_skills

        registry = create_default_registry(db_pool)
        selection = await select_skills(registry, ToolContext.CHAT, query="hello")

        available_names = {s.name for s in selection.available}
        assert {"core-memory", "research", "meeting-prep"} <= available_names
        # Catalog is a superset of the active selection
        assert {s.name for s in selection.skills} <= available_names


PLUGIN_SKILL_MD = """---
name: plugin-demo
description: Demo workflow provided by a plugin for testing
category: other
requires:
  tools: [recall]
contexts: [chat, heartbeat]
bound_tools: [recall]
---

# Plugin Demo

When the user asks about the plugin demo workflow, recall related context
first, then respond using the recalled evidence.
"""


class TestPluginSkillDirs:
    async def test_plugin_skill_dir_is_discoverable_and_activatable(
        self, db_pool, tmp_path
    ):
        from core.tools import (
            ToolContext,
            ToolExecutionContext,
            create_default_registry,
        )
        from core.tools.skills import ListSkillsHandler, UseSkillHandler
        from services.skill_runtime import get_skill_by_name, select_skills

        skill_dir = tmp_path / "plugin-demo"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(PLUGIN_SKILL_MD, encoding="utf-8")

        registry = create_default_registry(db_pool)
        registry.extra_skill_dirs = [tmp_path]

        # Present in the selection catalog (and thus the prompt index)
        selection = await select_skills(registry, ToolContext.CHAT, query="hello")
        assert "plugin-demo" in {s.name for s in selection.available}
        assert "queue_user_message" in selection.allowed_tool_names

        # Discoverable through list_skills
        ctx = ToolExecutionContext(
            tool_context=ToolContext.CHAT,
            call_id="plugin-skill-test",
            registry=registry,
        )
        listed = await ListSkillsHandler().execute({}, ctx)
        assert "plugin-demo" in {s["name"] for s in listed.output["skills"]}

        # Activatable through use_skill, unlocking its bound tools
        assert (
            await get_skill_by_name(registry, ToolContext.CHAT, "plugin-demo")
            is not None
        )
        activated = await UseSkillHandler().execute({"name": "plugin-demo"}, ctx)
        assert activated.success is True
        assert "Plugin Demo" in activated.output["instructions"]
        assert "recall" in activated.output["bound_tools"]

    async def test_create_full_registry_adopts_plugin_skill_dirs(
        self, db_pool, monkeypatch, tmp_path
    ):
        from core.tools.registry import create_full_registry

        class _StubPluginRegistry:
            def get_tool_handlers(self):
                return []

            def get_hooks(self):
                return []

            def get_skill_dirs(self):
                return [tmp_path]

        async def _fake_load_plugins(pool):
            return _StubPluginRegistry()

        monkeypatch.setattr("plugins.loader.load_plugins", _fake_load_plugins)
        registry = await create_full_registry(db_pool)
        assert registry.extra_skill_dirs == [tmp_path]


class TestSkillDiscoveryTools:
    async def test_list_skills_and_use_skill_return_discoverable_capabilities(
        self, db_pool
    ):
        from core.tools import (
            ToolContext,
            ToolExecutionContext,
            create_default_registry,
        )
        from core.tools.skills import ListSkillsHandler, UseSkillHandler

        registry = create_default_registry(db_pool)
        ctx = ToolExecutionContext(
            tool_context=ToolContext.CHAT,
            call_id="skill-test",
            registry=registry,
        )

        listed = await ListSkillsHandler().execute({}, ctx)
        assert listed.success is True
        names = {s["name"] for s in listed.output["skills"]}
        assert {"core-memory", "research"} <= names

        activated = await UseSkillHandler().execute({"name": "research"}, ctx)
        assert activated.success is True
        assert activated.output["name"] == "research"
        assert "Research Methodology" in activated.output["instructions"]
        assert {"web_search", "web_fetch"} <= set(activated.output["bound_tools"])

    async def test_author_skill_creates_parseable_agent_skill(
        self, db_pool, monkeypatch, tmp_path
    ):
        from core.tools import (
            ToolContext,
            ToolExecutionContext,
            create_default_registry,
        )
        from core.tools.skills import AuthorSkillHandler
        from skills.loader import load_skills_from_dir

        agent_root = tmp_path / "agent-authored"
        monkeypatch.setattr("core.tools.skills.AGENT_AUTHORED_SKILLS_DIR", agent_root)
        registry = create_default_registry(db_pool)
        ctx = ToolExecutionContext(
            tool_context=ToolContext.CHAT,
            call_id="author-skill-test",
            registry=registry,
        )

        result = await AuthorSkillHandler().execute(
            {
                "name": "weekly-review",
                "description": "Run a reusable weekly review workflow",
                "category": "productivity",
                "contexts": ["chat", "heartbeat"],
                "bound_tools": ["recall", "remember"],
                "content": (
                    "# Weekly Review\n\n"
                    "Use this when the user asks to review the week. First recall recent "
                    "commitments and goals, then summarize progress, blockers, and next "
                    "actions. Store durable decisions with remember and avoid inventing "
                    "events that were not recalled or provided in the current turn."
                ),
                "rationale": "The workflow is repeated and benefits from consistency.",
            },
            ctx,
        )

        assert result.success is True
        path = agent_root / "weekly-review" / "SKILL.md"
        assert path.exists()
        parsed = load_skills_from_dir(agent_root)
        assert len(parsed) == 1
        assert parsed[0].name == "weekly-review"
        assert parsed[0].bound_tools == ["recall", "remember"]
        assert parsed[0].provenance["authored_by"] == "hexis"
        assert parsed[0].provenance["managed_by"] == "author_skill"

    async def test_propose_skill_creates_pending_review_before_apply(
        self, db_pool, monkeypatch, tmp_path
    ):
        from core.tools import (
            ToolContext,
            ToolExecutionContext,
            create_default_registry,
        )
        from core.tools.skills import ProposeSkillHandler, ReviewSkillProposalHandler
        from skills.loader import load_skills_from_dir

        agent_root = tmp_path / "agent-authored"
        monkeypatch.setattr("core.tools.skills.AGENT_AUTHORED_SKILLS_DIR", agent_root)
        registry = create_default_registry(db_pool)
        ctx = ToolExecutionContext(
            tool_context=ToolContext.CHAT,
            call_id="propose-skill-test",
            session_id="11111111-1111-4111-8111-111111111111",
            registry=registry,
        )
        skill_name = f"inbox-triage-{uuid.uuid4().hex[:8]}"
        payload = {
            "need": "The agent noticed a reusable inbox triage workflow that is not covered by any current skill.",
            "name": skill_name,
            "description": "Triage an inbox for important actionable items",
            "category": "productivity",
            "contexts": ["chat", "heartbeat"],
            "bound_tools": ["recall"],
            "content": (
                "# Inbox Triage\n\n"
                "Use this when the user asks for recurring inbox triage. First inspect "
                "available connector state, then identify messages that are urgent, "
                "important, or waiting on the user. Summarize actions without sending, "
                "deleting, labeling, or notifying anyone unless a separate authorization "
                "grant explicitly allows that effect."
            ),
            "rationale": "Inbox triage is a reusable operational workflow and should not be rederived every time.",
            "confidence": 0.82,
        }

        result = await ProposeSkillHandler().execute(payload, ctx)

        assert result.success is True
        proposal_id = result.output["proposal_id"]
        assert result.output["writes_skill_file"] is False
        assert not (agent_root / skill_name / "SKILL.md").exists()
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT status, name, confidence, source_unit_ids, evidence
                FROM skill_improvement_proposals
                WHERE id = $1::uuid
                """,
                proposal_id,
            )
        assert row is not None
        assert row["status"] == "pending"
        assert row["name"] == skill_name
        assert row["confidence"] == pytest.approx(0.82)
        assert list(row["source_unit_ids"]) == []
        evidence = (
            row["evidence"]
            if isinstance(row["evidence"], dict)
            else json.loads(row["evidence"])
        )
        assert evidence["origin"] == "on_demand"
        assert evidence["call_id"] == "propose-skill-test"
        assert "inbox triage workflow" in evidence["need"]

        applied = await ReviewSkillProposalHandler().execute(
            {"proposal_id": proposal_id, "action": "apply"},
            ctx,
        )

        assert applied.success is True
        path = agent_root / skill_name / "SKILL.md"
        assert path.exists()
        parsed = load_skills_from_dir(agent_root)
        assert len(parsed) == 1
        assert parsed[0].name == skill_name
        assert parsed[0].provenance["proposal_id"] == proposal_id

    async def test_author_skill_refuses_accidental_overwrite(
        self, db_pool, monkeypatch, tmp_path
    ):
        from core.tools import (
            ToolContext,
            ToolExecutionContext,
            create_default_registry,
        )
        from core.tools.skills import AuthorSkillHandler

        monkeypatch.setattr(
            "core.tools.skills.AGENT_AUTHORED_SKILLS_DIR",
            tmp_path / "agent-authored",
        )
        registry = create_default_registry(db_pool)
        ctx = ToolExecutionContext(
            tool_context=ToolContext.CHAT,
            call_id="author-skill-test",
            registry=registry,
        )
        payload = {
            "name": "repeatable-method",
            "description": "Capture a repeatable method",
            "bound_tools": ["recall"],
            "content": (
                "# Repeatable Method\n\n"
                "Use this for a repeated procedure. Recall relevant context, follow "
                "the established steps, verify the result, and remember only durable "
                "lessons or decisions that will matter in future sessions."
            ),
        }

        first = await AuthorSkillHandler().execute(payload, ctx)
        second = await AuthorSkillHandler().execute(payload, ctx)

        assert first.success is True
        assert second.success is False
        assert "already exists" in (second.error or "")

    async def test_author_skill_updates_only_structured_agent_owned_skill(
        self, db_pool, monkeypatch, tmp_path
    ):
        from core.tools import (
            ToolContext,
            ToolExecutionContext,
            create_default_registry,
        )
        from core.tools.skills import AuthorSkillHandler
        from skills.loader import load_skills_from_dir

        agent_root = tmp_path / "agent-authored"
        monkeypatch.setattr("core.tools.skills.AGENT_AUTHORED_SKILLS_DIR", agent_root)
        registry = create_default_registry(db_pool)
        ctx = ToolExecutionContext(
            tool_context=ToolContext.CHAT,
            call_id="author-skill-update-test",
            registry=registry,
        )
        payload = {
            "name": "durable-review",
            "description": "Capture a durable review process",
            "content": (
                "# Durable Review\n\nRecall relevant evidence, separate observations from "
                "inferences, identify unresolved questions, and record only decisions "
                "that will remain useful across future sessions and changing context."
            ),
        }

        created = await AuthorSkillHandler().execute(payload, ctx)
        original = load_skills_from_dir(agent_root)[0]
        updated = await AuthorSkillHandler().execute(
            {
                **payload,
                "mode": "update",
                "description": "Run an evidence-based durable review process",
            },
            ctx,
        )
        revised = load_skills_from_dir(agent_root)[0]

        assert created.success is True
        assert updated.success is True
        assert revised.description == "Run an evidence-based durable review process"
        assert revised.provenance["created_at"] == original.provenance["created_at"]
        assert revised.provenance["managed_by"] == "author_skill"

    async def test_author_skill_refuses_unmarked_user_file_without_modifying_it(
        self, db_pool, monkeypatch, tmp_path
    ):
        from core.tools import (
            ToolContext,
            ToolErrorType,
            ToolExecutionContext,
            create_default_registry,
        )
        from core.tools.skills import AuthorSkillHandler

        agent_root = tmp_path / "agent-authored"
        monkeypatch.setattr("core.tools.skills.AGENT_AUTHORED_SKILLS_DIR", agent_root)
        path = agent_root / "personal-method" / "SKILL.md"
        path.parent.mkdir(parents=True)
        original = (
            "---\nname: personal-method\ndescription: A user-authored method\n---\n\n"
            "# Personal Method\n\nThis file belongs to the user and intentionally has no "
            "Hexis management marker. Its exact content must survive any attempted "
            "agent update, regardless of the requested replacement instructions.\n"
        )
        path.write_text(original, encoding="utf-8")
        registry = create_default_registry(db_pool)
        ctx = ToolExecutionContext(
            tool_context=ToolContext.CHAT,
            call_id="author-skill-ownership-test",
            registry=registry,
        )

        result = await AuthorSkillHandler().execute(
            {
                "name": "personal-method",
                "description": "Attempted replacement",
                "mode": "update",
                "content": (
                    "# Replacement\n\nThis replacement is long enough to satisfy content "
                    "validation, but it must never overwrite a file whose ownership "
                    "cannot be proven as agent-authored and managed by author_skill."
                ),
            },
            ctx,
        )

        assert result.success is False
        assert result.error_type is ToolErrorType.PERMISSION_DENIED
        assert "not marked as managed by Hexis" in (result.error or "")
        assert path.read_text(encoding="utf-8") == original

    async def test_author_skill_upgrades_legacy_agent_provenance_on_update(
        self, db_pool, monkeypatch, tmp_path
    ):
        from core.tools import (
            ToolContext,
            ToolExecutionContext,
            create_default_registry,
        )
        from core.tools.skills import AuthorSkillHandler
        from skills.loader import load_skills_from_dir

        agent_root = tmp_path / "agent-authored"
        monkeypatch.setattr("core.tools.skills.AGENT_AUTHORED_SKILLS_DIR", agent_root)
        path = agent_root / "legacy-method" / "SKILL.md"
        path.parent.mkdir(parents=True)
        path.write_text(
            "---\nname: legacy-method\ndescription: A legacy agent method\n---\n\n"
            "# Legacy Method\n\nKeep this established workflow usable while migrating "
            "its ownership metadata to the structured format. The historical footer "
            "is the only ownership evidence available for files from earlier Hexis versions.\n\n"
            "## Provenance\n\n- Authored by Hexis via `author_skill`.\n",
            encoding="utf-8",
        )
        registry = create_default_registry(db_pool)
        ctx = ToolExecutionContext(
            tool_context=ToolContext.CHAT,
            call_id="author-skill-legacy-test",
            registry=registry,
        )

        result = await AuthorSkillHandler().execute(
            {
                "name": "legacy-method",
                "description": "An upgraded legacy agent method",
                "mode": "update",
                "content": (
                    "# Legacy Method\n\nContinue the established workflow, verify each step "
                    "against current evidence, and preserve durable lessons. This approved "
                    "update also migrates ownership into structured frontmatter."
                ),
            },
            ctx,
        )
        parsed = load_skills_from_dir(agent_root)[0]

        assert result.success is True
        assert parsed.provenance["authored_by"] == "hexis"
        assert parsed.provenance["managed_by"] == "author_skill"

    async def test_author_skill_refuses_symlinked_target(
        self, db_pool, monkeypatch, tmp_path
    ):
        from core.tools import (
            ToolContext,
            ToolErrorType,
            ToolExecutionContext,
            create_default_registry,
        )
        from core.tools.skills import AuthorSkillHandler

        agent_root = tmp_path / "agent-authored"
        outside = tmp_path / "user-owned"
        outside.mkdir()
        agent_root.mkdir()
        (agent_root / "linked-method").symlink_to(outside, target_is_directory=True)
        monkeypatch.setattr("core.tools.skills.AGENT_AUTHORED_SKILLS_DIR", agent_root)
        registry = create_default_registry(db_pool)
        ctx = ToolExecutionContext(
            tool_context=ToolContext.CHAT,
            call_id="author-skill-symlink-test",
            registry=registry,
        )

        result = await AuthorSkillHandler().execute(
            {
                "name": "linked-method",
                "description": "Attempt a symlinked write",
                "content": (
                    "# Linked Method\n\nThis content is deliberately long enough for "
                    "validation, "
                    "but the target directory is a symlink outside the managed root and "
                    "must never receive an agent-authored skill document."
                ),
            },
            ctx,
        )

        assert result.success is False
        assert result.error_type is ToolErrorType.PATH_NOT_ALLOWED
        assert "may not be symlinks" in (result.error or "")
        assert not (outside / "SKILL.md").exists()

    async def test_author_skill_journals_and_notifies(
        self, db_pool, monkeypatch, tmp_path
    ):
        """Substrate-change visibility (#93/#99): authoring a skill journals a
        self_extension change and pins a first-person notice to the web inbox."""
        import json as jsonlib

        from core.tools import (
            ToolContext,
            ToolExecutionContext,
            create_default_registry,
        )
        from core.tools.skills import AuthorSkillHandler

        agent_root = tmp_path / "agent-authored"
        monkeypatch.setattr("core.tools.skills.AGENT_AUTHORED_SKILLS_DIR", agent_root)
        registry = create_default_registry(db_pool)
        ctx = ToolExecutionContext(
            tool_context=ToolContext.CHAT,
            call_id="author-skill-journal-test",
            registry=registry,
        )

        result = await AuthorSkillHandler().execute(
            {
                "name": "journaled-method",
                "description": "A method whose authoring must be visible to the operator",
                "bound_tools": ["recall"],
                "content": (
                    "# Journaled Method\n\nUse this to prove visibility: the act of "
                    "authoring this skill must land in the change journal and post a "
                    "first-person notice to the operator's web inbox, without blocking."
                ),
            },
            ctx,
        )
        assert result.success is True

        async with db_pool.acquire() as conn:
            try:
                journal = await conn.fetchrow(
                    """
                    SELECT summary, detail FROM change_journal
                    WHERE kind = 'self_extension' AND summary LIKE '%journaled-method%'
                    ORDER BY occurred_at DESC LIMIT 1
                    """
                )
                assert journal is not None
                assert "created" in journal["summary"]
                detail = jsonlib.loads(journal["detail"])
                assert detail["skill"] == "journaled-method"
                assert detail["mode"] == "create"
                assert detail["bound_tools"] == ["recall"]

                outbox = await conn.fetchrow(
                    """
                    SELECT envelope FROM outbox_messages
                    WHERE envelope->'payload'->>'intent' = 'self_extension'
                      AND envelope->'payload'->>'message' LIKE '%journaled-method%'
                    ORDER BY created_at DESC LIMIT 1
                    """
                )
                assert outbox is not None
                envelope = jsonlib.loads(outbox["envelope"])
                assert envelope["payload"]["delivery"] == {"mode": "web_inbox"}
                assert envelope["payload"]["message"].startswith("I wrote")
            finally:
                await conn.execute(
                    "DELETE FROM change_journal WHERE kind = 'self_extension' AND summary LIKE '%journaled-method%'"
                )
                await conn.execute(
                    """
                    DELETE FROM outbox_messages
                    WHERE envelope->'payload'->>'intent' = 'self_extension'
                      AND envelope->'payload'->>'message' LIKE '%journaled-method%'
                    """
                )


# ============================================================================
# J.1: SkillCategory Enum
# ============================================================================


class TestSkillCategory:
    def test_all_categories(self):
        expected = {
            "research",
            "productivity",
            "communication",
            "knowledge",
            "analytics",
            "creative",
            "system",
            "other",
        }
        actual = {c.value for c in SkillCategory}
        assert actual == expected

    def test_category_is_string_enum(self):
        assert SkillCategory.RESEARCH == "research"
        assert SkillCategory.ANALYTICS.value == "analytics"


# ============================================================================
# J.1: Requirement Checking
# ============================================================================


class TestRequirementChecking:
    def test_check_bins_available(self):
        """Python should always be available in test env."""
        spec = SkillSpec(
            name="test",
            description="test",
            content="test",
            requires_bins=["python3"],
        )
        missing = spec.check_bins_available()
        # python3 should be available
        assert "python3" not in missing

    def test_check_bins_missing(self):
        spec = SkillSpec(
            name="test",
            description="test",
            content="test",
            requires_bins=["nonexistent_binary_xyz_12345"],
        )
        missing = spec.check_bins_available()
        assert "nonexistent_binary_xyz_12345" in missing

    def test_check_env_available(self):
        with patch.dict(os.environ, {"TEST_HEXIS_VAR": "value"}):
            spec = SkillSpec(
                name="test",
                description="test",
                content="test",
                requires_env=["TEST_HEXIS_VAR"],
            )
            missing = spec.check_env_available()
            assert len(missing) == 0

    def test_check_env_missing(self):
        spec = SkillSpec(
            name="test",
            description="test",
            content="test",
            requires_env=["NONEXISTENT_ENV_VAR_XYZ"],
        )
        missing = spec.check_env_available()
        assert "NONEXISTENT_ENV_VAR_XYZ" in missing

    def test_check_os_support(self):
        import sys

        spec = SkillSpec(
            name="test",
            description="test",
            content="test",
            os_support=[sys.platform],
        )
        assert spec.check_os_support() is True

    def test_check_os_unsupported(self):
        spec = SkillSpec(
            name="test",
            description="test",
            content="test",
            os_support=["win32_only_platform"],
        )
        assert spec.check_os_support() is False

    def test_full_requirements_met_all_ok(self):
        import sys

        with patch.dict(os.environ, {"TEST_VAR": "1"}):
            spec = SkillSpec(
                name="test",
                description="test",
                content="test",
                requires_tools=["recall"],
                requires_env=["TEST_VAR"],
                requires_bins=["python3"],
                os_support=[sys.platform],
            )
            met, reasons = spec.full_requirements_met(
                available_tools={"recall"},
            )
            assert met is True
            assert len(reasons) == 0

    def test_full_requirements_met_disabled(self):
        spec = SkillSpec(
            name="test",
            description="test",
            content="test",
            enabled=False,
        )
        met, reasons = spec.full_requirements_met(available_tools=set())
        assert met is False
        assert any("disabled" in r for r in reasons)

    def test_full_requirements_met_missing_tool(self):
        spec = SkillSpec(
            name="test",
            description="test",
            content="test",
            requires_tools=["nonexistent_tool"],
        )
        met, reasons = spec.full_requirements_met(available_tools=set())
        assert met is False
        assert any("tool" in r for r in reasons)

    def test_full_requirements_met_missing_env(self):
        spec = SkillSpec(
            name="test",
            description="test",
            content="test",
            requires_env=["NONEXISTENT_VAR_ABC"],
        )
        met, reasons = spec.full_requirements_met(available_tools=set())
        assert met is False
        assert any("env var" in r for r in reasons)


# ============================================================================
# J.2: Built-in Skill Library
# ============================================================================


class TestBuiltInSkillLibrary:
    def test_bundled_skills_load(self):
        skills_dir = Path(__file__).resolve().parents[2] / "skills" / "installed"
        skills = load_skills_from_dir(skills_dir)
        names = [s.name for s in skills]
        # Should have at least research + self-reflection + new skills
        assert "research" in names
        assert "self-reflection" in names
        assert len(skills) >= 4  # At least 4 skills total

    def test_new_skills_have_categories(self):
        skills_dir = Path(__file__).resolve().parents[2] / "skills" / "installed"
        skills = load_skills_from_dir(skills_dir)
        for skill in skills:
            if skill.name not in ("research", "self-reflection"):
                # New skills should have a non-OTHER category
                assert (
                    skill.category != SkillCategory.OTHER or skill.name == "research"
                ), f"Skill {skill.name} should have a specific category"

    def test_new_skills_have_bound_tools(self):
        skills_dir = Path(__file__).resolve().parents[2] / "skills" / "installed"
        skills = load_skills_from_dir(skills_dir)
        skills_with_tools = [s for s in skills if s.bound_tools]
        assert len(skills_with_tools) >= 4

    def test_all_skills_parse_correctly(self):
        skills_dir = Path(__file__).resolve().parents[2] / "skills" / "installed"
        skills = load_skills_from_dir(skills_dir)
        for skill in skills:
            assert skill.name, f"Skill from {skill.source} has no name"
            assert skill.description, f"Skill {skill.name} has no description"
            assert skill.content, f"Skill {skill.name} has no content"
            assert len(skill.contexts) > 0, f"Skill {skill.name} has no contexts"

    def test_skill_content_is_substantive(self):
        """Skills should have meaningful content (at least 100 chars)."""
        skills_dir = Path(__file__).resolve().parents[2] / "skills" / "installed"
        skills = load_skills_from_dir(skills_dir)
        for skill in skills:
            assert len(skill.content) >= 100, (
                f"Skill {skill.name} content too short ({len(skill.content)} chars)"
            )


# ============================================================================
# J.3: Skill Installer
# ============================================================================


class TestSkillInstaller:
    def test_install_no_deps(self):
        spec = SkillSpec(name="test", description="test", content="test")
        results = install_skill_deps(spec)
        assert len(results) == 0

    def test_install_unsupported_os(self):
        spec = SkillSpec(
            name="test",
            description="test",
            content="test",
            os_support=["win32_only"],
        )
        results = install_skill_deps(spec)
        assert len(results) == 1
        assert results[0]["status"] == "unsupported"

    def test_install_reports_missing_env(self):
        spec = SkillSpec(
            name="test",
            description="test",
            content="test",
            requires_env=["NONEXISTENT_TEST_VAR_XYZ"],
        )
        results = install_skill_deps(spec)
        assert len(results) == 1
        assert results[0]["status"] == "missing"
        assert "env var" in results[0]["detail"]

    def test_install_reports_present_env(self):
        with patch.dict(os.environ, {"PRESENT_VAR": "value"}):
            spec = SkillSpec(
                name="test",
                description="test",
                content="test",
                requires_env=["PRESENT_VAR"],
            )
            results = install_skill_deps(spec)
            assert len(results) == 1
            assert results[0]["status"] == "ok"

    def test_install_bins_already_present(self):
        spec = SkillSpec(
            name="test",
            description="test",
            content="test",
            requires_bins=["python3"],
        )
        results = install_skill_deps(spec)
        assert len(results) == 1
        assert results[0]["status"] == "ok"
        assert "already installed" in results[0]["detail"]

    def test_install_missing_bins_no_method(self):
        spec = SkillSpec(
            name="test",
            description="test",
            content="test",
            requires_bins=["nonexistent_bin_xyz"],
        )
        results = install_skill_deps(spec)
        assert any(r["status"] == "missing" for r in results)

    @patch("skills.loader.shutil.which")
    @patch("skills.loader._run_install_command")
    def test_install_with_brew(self, mock_run, mock_which):
        mock_which.side_effect = lambda x: (
            "/usr/local/bin/brew" if x == "brew" else None
        )
        mock_run.return_value = (True, "installed")

        spec = SkillSpec(
            name="test",
            description="test",
            content="test",
            requires_bins=["rg"],
            install_methods=[
                InstallMethod(kind="brew", package="ripgrep", bins=["rg"])
            ],
        )
        results = install_skill_deps(spec)
        installed = [r for r in results if r["status"] == "installed"]
        assert len(installed) == 1
        mock_run.assert_called_once()

    @patch("skills.loader.shutil.which")
    def test_install_installer_not_available(self, mock_which):
        mock_which.return_value = None  # No brew available

        spec = SkillSpec(
            name="test",
            description="test",
            content="test",
            requires_bins=["nonexistent_tool"],
            install_methods=[
                InstallMethod(
                    kind="brew", package="some-pkg", bins=["nonexistent_tool"]
                )
            ],
        )
        results = install_skill_deps(spec)
        skipped = [r for r in results if r["status"] == "skipped"]
        assert len(skipped) >= 1

    @patch("skills.loader.shutil.which")
    @patch("skills.loader._run_install_command")
    def test_install_failure(self, mock_run, mock_which):
        mock_which.side_effect = lambda x: "/usr/bin/pip" if x == "pip" else None
        mock_run.return_value = (False, "pip install failed")

        spec = SkillSpec(
            name="test",
            description="test",
            content="test",
            requires_bins=["some_binary"],
            install_methods=[
                InstallMethod(kind="pip", package="some-pkg", bins=["some_binary"])
            ],
        )
        results = install_skill_deps(spec)
        failed = [r for r in results if r["status"] == "failed"]
        assert len(failed) == 1


# ============================================================================
# J.4: Skill-to-Tool Binding
# ============================================================================


class TestSkillToolBinding:
    def test_bound_tools_in_spec(self):
        """Skills can declare bound_tools."""
        spec = SkillSpec(
            name="email-digest",
            description="Email digest skill",
            content="content",
            bound_tools=["list_emails", "read_email", "ingest_emails"],
        )
        assert len(spec.bound_tools) == 3

    def test_bound_tools_from_frontmatter(self):
        metadata = {
            "name": "test",
            "description": "test",
            "bound_tools": ["web_search", "recall"],
        }
        spec = SkillSpec.from_frontmatter(metadata, "content")
        assert spec.bound_tools == ["web_search", "recall"]

    def test_bound_tools_in_skill_file(self):
        """Test that bundled skills with bound_tools parse correctly."""
        skills_dir = Path(__file__).resolve().parents[2] / "skills" / "installed"
        skills = load_skills_from_dir(skills_dir)
        bound = {s.name: s.bound_tools for s in skills if s.bound_tools}
        # At least some skills should have bound_tools
        assert len(bound) >= 1

    def test_to_prompt_block_includes_content(self):
        """Prompt block should include the skill content for context."""
        spec = SkillSpec(
            name="test",
            description="test skill",
            content="# Test\nDo the thing.",
            bound_tools=["recall"],
        )
        block = spec.to_prompt_block()
        assert '<skill name="test">' in block
        assert "# Test" in block
        assert "</skill>" in block


# ============================================================================
# Discover Skill Dirs
# ============================================================================


class TestDiscoverSkillDirs:
    def test_discovers_bundled(self):
        dirs = discover_skill_dirs()
        assert any(str(d).endswith("installed") for d in dirs)

    def test_user_dir_included_if_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("skills.loader._USER_SKILLS_DIR", Path(tmpdir)):
                dirs = discover_skill_dirs()
                assert Path(tmpdir) in dirs

    def test_user_dir_excluded_if_missing(self):
        with patch("skills.loader._USER_SKILLS_DIR", Path("/nonexistent/path/xyz")):
            dirs = discover_skill_dirs()
            assert Path("/nonexistent/path/xyz") not in dirs


# ============================================================================
# Load Skills with Extended Filtering
# ============================================================================


class TestLoadSkillsExtended:
    def test_load_with_extra_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "custom_skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "---\nname: custom\ndescription: custom skill\n"
                "category: system\ncontexts: [chat]\n"
                "bound_tools: [recall]\n---\n# Custom\nDo things.",
                encoding="utf-8",
            )

            skills = load_skills(
                context=SkillContext.CHAT,
                available_tools=set(),
                extra_dirs=[Path(tmpdir)],
            )
            names = [s.name for s in skills]
            assert "custom" in names

    def test_deduplication_user_over_bundled(self):
        """User skills with same name should override bundled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "research"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "---\nname: research\ndescription: custom research\n"
                "contexts: [chat]\n---\n# Custom Research\nMy version.",
                encoding="utf-8",
            )

            with patch("skills.loader._USER_SKILLS_DIR", Path(tmpdir)):
                skills = load_skills(
                    context=SkillContext.CHAT,
                    available_tools={"web_search", "web_fetch"},
                    available_config={"tavily"},
                )
                research = [s for s in skills if s.name == "research"]
                assert len(research) == 1
                assert "custom" in research[0].description.lower()
