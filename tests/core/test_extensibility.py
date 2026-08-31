"""
Tests for the extensibility system: optional tools, hooks, skills, plugins.

Covers Phases A-D of the OpenClaw-inspired extensibility proposal.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


# ============================================================================
# Phase A: Optional Tools
# ============================================================================


class TestOptionalToolSpec:
    """ToolSpec.optional flag and policy enforcement."""

    def test_default_not_optional(self):
        from core.tools.base import ToolSpec, ToolCategory

        spec = ToolSpec(
            name="test_tool",
            description="test",
            parameters={"type": "object", "properties": {}},
            category=ToolCategory.MEMORY,
        )
        assert spec.optional is False

    def test_optional_flag(self):
        from core.tools.base import ToolSpec, ToolCategory

        spec = ToolSpec(
            name="test_tool",
            description="test",
            parameters={"type": "object", "properties": {}},
            category=ToolCategory.MEMORY,
            optional=True,
        )
        assert spec.optional is True


class TestOptionalToolConfig:
    """ToolsConfig allowlist for optional tools."""

    def test_empty_allowlist_denies(self):
        from core.tools.config import ToolsConfig
        from core.tools.base import ToolCategory

        config = ToolsConfig()
        assert config.is_optional_allowed("weather", ToolCategory.EXTERNAL) is False

    def test_tool_name_in_allowlist(self):
        from core.tools.config import ToolsConfig
        from core.tools.base import ToolCategory

        config = ToolsConfig(allowed_optional=["weather"])
        assert config.is_optional_allowed("weather", ToolCategory.EXTERNAL) is True
        assert config.is_optional_allowed("other", ToolCategory.EXTERNAL) is False

    def test_category_group_in_allowlist(self):
        from core.tools.config import ToolsConfig
        from core.tools.base import ToolCategory

        config = ToolsConfig(allowed_optional_groups=["external"])
        assert config.is_optional_allowed("weather", ToolCategory.EXTERNAL) is True
        assert config.is_optional_allowed("something", ToolCategory.MEMORY) is False

    def test_plugins_group_allows_all(self):
        from core.tools.config import ToolsConfig
        from core.tools.base import ToolCategory

        config = ToolsConfig(allowed_optional_groups=["plugins"])
        assert config.is_optional_allowed("anything", ToolCategory.EXTERNAL) is True
        assert config.is_optional_allowed("anything", ToolCategory.WEB) is True

    def test_roundtrip_json(self):
        from core.tools.config import ToolsConfig

        config = ToolsConfig(
            allowed_optional=["weather", "home_assistant"],
            allowed_optional_groups=["external"],
        )
        data = config.to_dict()
        restored = ToolsConfig.from_json(data)
        assert restored.allowed_optional == ["weather", "home_assistant"]
        assert restored.allowed_optional_groups == ["external"]

    def test_from_json_missing_fields(self):
        from core.tools.config import ToolsConfig

        config = ToolsConfig.from_json({"enabled": None})
        assert config.allowed_optional == []
        assert config.allowed_optional_groups == []


class TestOptionalToolPolicy:
    """Policy enforcement of optional tools."""

    async def test_optional_denied_without_allowlist(self, db_pool):
        from core.tools.base import ToolSpec, ToolCategory, ToolContext
        from core.tools.config import ToolsConfig
        from core.tools.policy import ToolPolicy

        spec = ToolSpec(
            name="weather",
            description="test",
            parameters={"type": "object", "properties": {}},
            category=ToolCategory.EXTERNAL,
            optional=True,
        )
        policy = ToolPolicy(db_pool)
        config = ToolsConfig()

        result = await policy.check_all(
            spec=spec,
            context=ToolContext.CHAT,
            config=config,
        )
        assert result.allowed is False
        assert "allowlist" in (result.reason or "")

    async def test_optional_allowed_with_allowlist(self, db_pool):
        from core.tools.base import ToolSpec, ToolCategory, ToolContext
        from core.tools.config import ToolsConfig
        from core.tools.policy import ToolPolicy

        spec = ToolSpec(
            name="weather",
            description="test",
            parameters={"type": "object", "properties": {}},
            category=ToolCategory.EXTERNAL,
            optional=True,
        )
        policy = ToolPolicy(db_pool)
        config = ToolsConfig(allowed_optional=["weather"])

        result = await policy.check_all(
            spec=spec,
            context=ToolContext.CHAT,
            config=config,
        )
        assert result.allowed is True

    async def test_non_optional_unaffected(self, db_pool):
        from core.tools.base import ToolSpec, ToolCategory, ToolContext
        from core.tools.config import ToolsConfig
        from core.tools.policy import ToolPolicy

        spec = ToolSpec(
            name="recall",
            description="test",
            parameters={"type": "object", "properties": {}},
            category=ToolCategory.MEMORY,
        )
        policy = ToolPolicy(db_pool)
        config = ToolsConfig()

        result = await policy.check_all(
            spec=spec,
            context=ToolContext.CHAT,
            config=config,
        )
        assert result.allowed is True


# ============================================================================
# Phase B: Hook System
# ============================================================================


class TestHookRegistry:
    """Core hook registry behavior."""

    def test_empty_registry(self):
        from core.tools.hooks import HookRegistry, HookEvent

        registry = HookRegistry()
        assert registry.count() == 0
        assert registry.count(HookEvent.BEFORE_TOOL_CALL) == 0

    async def test_register_and_run(self):
        from core.tools.hooks import (
            HookRegistry, HookEvent, HookContext, HookOutcome, FunctionHookHandler,
        )

        registry = HookRegistry()
        called = []

        async def my_hook(ctx: HookContext) -> HookOutcome | None:
            called.append(ctx.tool_name)
            return None

        registry.register_function(HookEvent.BEFORE_TOOL_CALL, my_hook, source="test")
        assert registry.count(HookEvent.BEFORE_TOOL_CALL) == 1

        outcome = await registry.run(
            HookEvent.BEFORE_TOOL_CALL,
            HookContext(event=HookEvent.BEFORE_TOOL_CALL, tool_name="recall"),
        )
        assert called == ["recall"]
        assert outcome.block is False

    async def test_hook_can_block(self):
        from core.tools.hooks import HookRegistry, HookEvent, HookContext, HookOutcome

        registry = HookRegistry()

        async def blocking_hook(ctx: HookContext) -> HookOutcome:
            return HookOutcome.blocked("not allowed")

        registry.register_function(HookEvent.BEFORE_TOOL_CALL, blocking_hook, source="test")

        outcome = await registry.run(
            HookEvent.BEFORE_TOOL_CALL,
            HookContext(event=HookEvent.BEFORE_TOOL_CALL, tool_name="recall"),
        )
        assert outcome.block is True
        assert outcome.block_reason == "not allowed"

    async def test_hook_can_mutate_arguments(self):
        from core.tools.hooks import HookRegistry, HookEvent, HookContext, HookOutcome

        registry = HookRegistry()

        async def mutating_hook(ctx: HookContext) -> HookOutcome:
            args = dict(ctx.arguments or {})
            args["injected"] = True
            return HookOutcome.with_args(args)

        registry.register_function(HookEvent.BEFORE_TOOL_CALL, mutating_hook, source="test")

        outcome = await registry.run(
            HookEvent.BEFORE_TOOL_CALL,
            HookContext(
                event=HookEvent.BEFORE_TOOL_CALL,
                tool_name="recall",
                arguments={"query": "test"},
            ),
        )
        assert outcome.mutated_arguments == {"query": "test", "injected": True}

    async def test_hook_priority_ordering(self):
        from core.tools.hooks import HookRegistry, HookEvent, HookContext, HookOutcome

        registry = HookRegistry()
        order = []

        async def hook_a(ctx: HookContext) -> HookOutcome | None:
            order.append("a")
            return None

        async def hook_b(ctx: HookContext) -> HookOutcome | None:
            order.append("b")
            return None

        # b has lower priority (runs first)
        registry.register_function(HookEvent.AFTER_TOOL_CALL, hook_a, source="test", priority=200)
        registry.register_function(HookEvent.AFTER_TOOL_CALL, hook_b, source="test", priority=50)

        await registry.run(
            HookEvent.AFTER_TOOL_CALL,
            HookContext(event=HookEvent.AFTER_TOOL_CALL),
        )
        assert order == ["b", "a"]

    async def test_hook_context_injection(self):
        from core.tools.hooks import HookRegistry, HookEvent, HookContext, HookOutcome

        registry = HookRegistry()

        async def prepend_hook(ctx: HookContext) -> HookOutcome:
            return HookOutcome(prepend_context="<context>relevant info</context>")

        async def append_hook(ctx: HookContext) -> HookOutcome:
            return HookOutcome(append_context="<footer>extra</footer>")

        registry.register_function(HookEvent.BEFORE_HEARTBEAT, prepend_hook, source="test")
        registry.register_function(HookEvent.BEFORE_HEARTBEAT, append_hook, source="test")

        outcome = await registry.run(
            HookEvent.BEFORE_HEARTBEAT,
            HookContext(event=HookEvent.BEFORE_HEARTBEAT),
        )
        assert outcome.prepend_context == "<context>relevant info</context>"
        assert outcome.append_context == "<footer>extra</footer>"

    async def test_hook_failure_doesnt_crash(self):
        from core.tools.hooks import HookRegistry, HookEvent, HookContext, HookOutcome

        registry = HookRegistry()

        async def crashing_hook(ctx: HookContext) -> HookOutcome:
            raise RuntimeError("hook exploded")

        async def ok_hook(ctx: HookContext) -> HookOutcome:
            return HookOutcome(metadata={"ran": True})

        registry.register_function(HookEvent.AFTER_TOOL_CALL, crashing_hook, source="bad")
        registry.register_function(HookEvent.AFTER_TOOL_CALL, ok_hook, source="good")

        outcome = await registry.run(
            HookEvent.AFTER_TOOL_CALL,
            HookContext(event=HookEvent.AFTER_TOOL_CALL),
        )
        # ok_hook still ran despite crashing_hook
        assert outcome.metadata.get("ran") is True

    def test_unregister_all(self):
        from core.tools.hooks import HookRegistry, HookEvent, HookContext, HookOutcome

        registry = HookRegistry()

        async def dummy(ctx):
            return None

        registry.register_function(HookEvent.BEFORE_TOOL_CALL, dummy, source="plugin_a")
        registry.register_function(HookEvent.AFTER_TOOL_CALL, dummy, source="plugin_a")
        registry.register_function(HookEvent.BEFORE_TOOL_CALL, dummy, source="plugin_b")

        assert registry.count() == 3
        removed = registry.unregister_all("plugin_a")
        assert removed == 2
        assert registry.count() == 1

    def test_list_hooks(self):
        from core.tools.hooks import HookRegistry, HookEvent

        registry = HookRegistry()

        async def dummy(ctx):
            return None

        registry.register_function(HookEvent.BEFORE_TOOL_CALL, dummy, source="test", name="my_hook")
        hooks = registry.list_hooks(HookEvent.BEFORE_TOOL_CALL)
        assert len(hooks) == 1
        assert hooks[0]["source"] == "test"
        assert hooks[0]["event"] == "before_tool_call"


# ============================================================================
# Phase C: Skills System
# ============================================================================


class TestSkillSpec:
    """SkillSpec parsing and methods."""

    def test_from_frontmatter(self):
        from skills.base import SkillSpec, SkillContext

        metadata = {
            "name": "research",
            "description": "Research methodology",
            "requires": {
                "tools": ["web_search", "web_fetch"],
                "config": ["tavily"],
            },
            "contexts": ["heartbeat", "chat"],
            "blueprint": {
                "title": "Weekly research prompt",
                "rationale": "A recurring prompt keeps the research review visible.",
                "schedule": "weekly:friday:16:00",
                "message": "Open Hexis for the weekly research review.",
            },
        }
        content = "# Research\n\nUse web_search to find things."

        spec = SkillSpec.from_frontmatter(metadata, content, source="test")
        assert spec.name == "research"
        assert spec.description == "Research methodology"
        assert spec.requires_tools == ["web_search", "web_fetch"]
        assert spec.requires_config == ["tavily"]
        assert SkillContext.HEARTBEAT in spec.contexts
        assert SkillContext.CHAT in spec.contexts
        assert spec.source == "test"
        assert spec.blueprint is not None
        assert spec.blueprint["schedule"] == "weekly:friday:16:00"

    def test_requirements_met(self):
        from skills.base import SkillSpec

        spec = SkillSpec(
            name="test",
            description="test",
            content="test",
            requires_tools=["web_search", "recall"],
            requires_config=["tavily"],
        )
        # All met
        assert spec.requirements_met(
            available_tools={"web_search", "recall", "remember"},
            available_config={"tavily"},
        ) is True

        # Missing tool
        assert spec.requirements_met(
            available_tools={"recall"},
            available_config={"tavily"},
        ) is False

        # Missing config
        assert spec.requirements_met(
            available_tools={"web_search", "recall"},
            available_config=set(),
        ) is False

    def test_requirements_met_no_config_check(self):
        from skills.base import SkillSpec

        spec = SkillSpec(
            name="test",
            description="test",
            content="test",
            requires_tools=["recall"],
            requires_config=["tavily"],
        )
        # When available_config is None, config check is skipped
        assert spec.requirements_met(
            available_tools={"recall"},
            available_config=None,
        ) is True

    def test_to_prompt_block(self):
        from skills.base import SkillSpec

        spec = SkillSpec(name="research", description="test", content="# Research\nDo things.")
        block = spec.to_prompt_block()
        assert '<skill name="research">' in block
        assert "# Research" in block
        assert "</skill>" in block


class TestSkillLoader:
    """Skill discovery and loading from directories."""

    def test_load_bundled_research_skill(self):
        from skills.loader import load_skills_from_dir
        from pathlib import Path

        skills_dir = Path(__file__).resolve().parents[2] / "skills" / "installed"
        skills = load_skills_from_dir(skills_dir)
        names = [s.name for s in skills]
        assert "research" in names
        assert "self-reflection" in names

    def test_load_skills_with_context_filter(self):
        from skills import load_skills
        from skills.base import SkillContext

        # self-reflection requires only recall+remember and heartbeat context
        skills = load_skills(
            context=SkillContext.HEARTBEAT,
            available_tools={"recall", "remember"},
            available_config=set(),  # No config keys available
        )
        names = [s.name for s in skills]
        assert "self-reflection" in names
        # research requires web_search+web_fetch+tavily config which aren't available
        assert "research" not in names

    def test_load_skills_requirements_filter(self):
        from skills import load_skills
        from skills.base import SkillContext

        # With web tools available, research should load
        skills = load_skills(
            context=SkillContext.CHAT,
            available_tools={"recall", "remember", "web_search", "web_fetch"},
            available_config={"tavily"},
        )
        names = [s.name for s in skills]
        assert "research" in names

    def test_load_from_nonexistent_dir(self):
        from skills.loader import load_skills_from_dir

        skills = load_skills_from_dir(Path("/nonexistent/dir"))
        assert skills == []

    def test_load_from_tempdir_with_skill(self):
        from skills.loader import load_skills_from_dir
        from skills.base import SkillContext

        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "my_skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "---\nname: my_test_skill\ndescription: test\n"
                "contexts: [chat]\n---\n# Test Skill\nHello.",
                encoding="utf-8",
            )

            skills = load_skills_from_dir(Path(tmpdir))
            assert len(skills) == 1
            assert skills[0].name == "my_test_skill"
            assert SkillContext.CHAT in skills[0].contexts

    def test_skill_without_frontmatter_skipped(self):
        from skills.loader import load_skills_from_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "bad_skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("# No Frontmatter\nJust text.", encoding="utf-8")

            skills = load_skills_from_dir(Path(tmpdir))
            assert len(skills) == 0


# ============================================================================
# Phase D: Plugin System
# ============================================================================


class TestPluginManifest:
    """PluginManifest parsing."""

    def test_from_dict(self):
        from plugins.base import PluginManifest

        manifest = PluginManifest.from_dict({
            "id": "weather",
            "name": "Weather Lookup",
            "version": "1.0.0",
            "description": "Current weather",
            "config_schema": {"type": "object", "properties": {"api_key": {"type": "string"}}},
        })
        assert manifest.id == "weather"
        assert manifest.name == "Weather Lookup"
        assert manifest.version == "1.0.0"
        assert "api_key" in manifest.config_schema.get("properties", {})

    def test_roundtrip(self):
        from plugins.base import PluginManifest

        original = PluginManifest(
            id="test", name="Test", version="2.0.0", description="A test plugin",
        )
        data = original.to_dict()
        restored = PluginManifest.from_dict(data)
        assert restored.id == original.id
        assert restored.version == original.version

    def test_from_json_file(self):
        from plugins.base import PluginManifest

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"id": "file_test", "name": "File Test", "version": "0.1.0"}, f)
            f.flush()
            manifest = PluginManifest.from_json_file(Path(f.name))
            assert manifest.id == "file_test"


class TestHexisPluginApi:
    """Plugin API registration methods."""

    def test_register_tool(self):
        from plugins.base import HexisPluginApi
        from core.tools.base import ToolHandler, ToolSpec, ToolCategory, ToolResult, ToolExecutionContext

        class DummyHandler(ToolHandler):
            @property
            def spec(self):
                return ToolSpec(
                    name="dummy",
                    description="test",
                    parameters={"type": "object", "properties": {}},
                    category=ToolCategory.EXTERNAL,
                )
            async def execute(self, arguments, context):
                return ToolResult.success_result("ok")

        pool = MagicMock()
        api = HexisPluginApi("test_plugin", pool)
        api.register_tool(DummyHandler())
        assert len(api._get_tools()) == 1
        assert api._get_tools()[0].handler.spec.name == "dummy"
        assert api._get_tools()[0].optional is False

    def test_register_optional_tool(self):
        from plugins.base import HexisPluginApi
        from core.tools.base import ToolHandler, ToolSpec, ToolCategory, ToolResult

        class DummyHandler(ToolHandler):
            @property
            def spec(self):
                return ToolSpec(
                    name="optional_dummy",
                    description="test",
                    parameters={"type": "object", "properties": {}},
                    category=ToolCategory.EXTERNAL,
                )
            async def execute(self, arguments, context):
                return ToolResult.success_result("ok")

        pool = MagicMock()
        api = HexisPluginApi("test_plugin", pool)
        api.register_tool(DummyHandler(), optional=True)
        assert api._get_tools()[0].optional is True
        assert api._get_tools()[0].handler.spec.optional is True

    def test_register_hook(self):
        from plugins.base import HexisPluginApi
        from core.tools.hooks import HookEvent, HookHandler, HookContext, HookOutcome

        class DummyHook(HookHandler):
            async def handle(self, context):
                return None

        pool = MagicMock()
        api = HexisPluginApi("test_plugin", pool)
        api.register_hook(HookEvent.BEFORE_TOOL_CALL, DummyHook())
        assert len(api._get_hooks()) == 1

    def test_register_skill_dir(self):
        from plugins.base import HexisPluginApi

        with tempfile.TemporaryDirectory() as tmpdir:
            pool = MagicMock()
            api = HexisPluginApi("test_plugin", pool)
            api.register_skill_dir(Path(tmpdir))
            assert len(api._get_skill_dirs()) == 1

    def test_register_skill_dir_nonexistent_ignored(self):
        from plugins.base import HexisPluginApi

        pool = MagicMock()
        api = HexisPluginApi("test_plugin", pool)
        api.register_skill_dir(Path("/nonexistent"))
        assert len(api._get_skill_dirs()) == 0


class TestPluginRegistry:
    """PluginRegistry aggregation."""

    def test_empty_registry(self):
        from plugins.registry import PluginRegistry

        registry = PluginRegistry()
        assert registry.plugin_count() == 0
        assert registry.tool_count() == 0
        assert registry.hook_count() == 0
        assert registry.get_tool_handlers() == []
        assert registry.get_hooks() == []

    def test_add_plugin(self):
        from plugins.registry import PluginRegistry, _PluginToolEntry, _PluginHookEntry
        from core.tools.base import ToolHandler, ToolSpec, ToolCategory, ToolResult
        from core.tools.hooks import HookEvent, HookHandler

        class DummyHandler(ToolHandler):
            @property
            def spec(self):
                return ToolSpec(
                    name="test_tool",
                    description="test",
                    parameters={"type": "object", "properties": {}},
                    category=ToolCategory.EXTERNAL,
                )
            async def execute(self, arguments, context):
                return ToolResult.success_result("ok")

        class DummyHook(HookHandler):
            async def handle(self, context):
                return None

        registry = PluginRegistry()
        handler = DummyHandler()
        hook = DummyHook()

        registry._add_plugin(
            plugin_id="test",
            manifest_dict={"id": "test", "name": "Test Plugin"},
            tools=[_PluginToolEntry(plugin_id="test", handler=handler, optional=False)],
            hooks=[_PluginHookEntry(plugin_id="test", event=HookEvent.AFTER_TOOL_CALL, handler=hook)],
            skill_dirs=[],
        )

        assert registry.plugin_count() == 1
        assert registry.tool_count() == 1
        assert registry.hook_count() == 1
        assert registry.get_tool_handlers()[0].spec.name == "test_tool"
        assert registry.list_plugins()[0]["id"] == "test"


class TestPluginDiscovery:
    """Plugin filesystem discovery."""

    def test_discover_empty_dir(self):
        from plugins.loader import discover_plugins

        with tempfile.TemporaryDirectory() as tmpdir:
            plugins = discover_plugins(extra_dirs=[Path(tmpdir)], include_bundled=False)
            # Only finds plugins with __init__.py
            assert len([p for p in plugins if str(tmpdir) in str(p)]) == 0

    def test_discover_valid_plugin(self):
        from plugins.loader import discover_plugins

        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir) / "my_plugin"
            plugin_dir.mkdir()
            (plugin_dir / "__init__.py").write_text("# plugin\n")
            (plugin_dir / "plugin.json").write_text('{"id": "my_plugin", "name": "My Plugin"}')

            plugins = discover_plugins(extra_dirs=[Path(tmpdir)], include_bundled=False)
            found = [p for p in plugins if "my_plugin" in str(p)]
            assert len(found) == 1

    def test_discover_skips_hidden_dirs(self):
        from plugins.loader import discover_plugins

        with tempfile.TemporaryDirectory() as tmpdir:
            hidden = Path(tmpdir) / ".hidden_plugin"
            hidden.mkdir()
            (hidden / "__init__.py").write_text("# plugin\n")

            plugins = discover_plugins(extra_dirs=[Path(tmpdir)], include_bundled=False)
            found = [p for p in plugins if ".hidden_plugin" in str(p)]
            assert len(found) == 0


class TestPluginLoading:
    """Full plugin loading integration."""

    async def test_load_plugin_with_tool(self, db_pool):
        from plugins.loader import load_plugins
        from plugins.base import HexisPlugin, PluginManifest, HexisPluginApi
        from core.tools.base import ToolHandler, ToolSpec, ToolCategory, ToolResult

        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir) / "sample"
            plugin_dir.mkdir()

            # Write a plugin that registers a tool
            (plugin_dir / "__init__.py").write_text(
                '''
from plugins.base import HexisPlugin, PluginManifest, HexisPluginApi
from core.tools.base import ToolHandler, ToolSpec, ToolCategory, ToolResult, ToolExecutionContext
from typing import Any

class SampleToolHandler(ToolHandler):
    @property
    def spec(self):
        return ToolSpec(
            name="sample_tool",
            description="A sample tool from a plugin",
            parameters={"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]},
            category=ToolCategory.EXTERNAL,
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        return ToolResult.success_result({"echo": arguments["input"]})


class SamplePlugin(HexisPlugin):
    @property
    def manifest(self):
        return PluginManifest(id="sample", name="Sample Plugin", version="1.0.0")

    def register(self, api: HexisPluginApi):
        api.register_tool(SampleToolHandler(), optional=True)

plugin = SamplePlugin()
''',
                encoding="utf-8",
            )

            registry = await load_plugins(db_pool, extra_dirs=[Path(tmpdir)], include_bundled=False)
            assert registry.plugin_count() == 1
            handlers = registry.get_tool_handlers()
            assert len(handlers) == 1
            assert handlers[0].spec.name == "sample_tool"
            assert handlers[0].spec.optional is True


# ============================================================================
# Phase B+D Integration: Hooks in Registry
# ============================================================================


class TestRegistryHookIntegration:
    """Hooks integrated into ToolRegistry.execute()."""

    async def test_hook_blocks_tool_execution(self, db_pool):
        from core.tools.registry import ToolRegistry
        from core.tools.base import (
            ToolHandler, ToolSpec, ToolCategory, ToolContext,
            ToolExecutionContext, ToolResult, ToolErrorType,
        )
        from core.tools.hooks import HookEvent, HookContext, HookOutcome

        class DummyHandler(ToolHandler):
            @property
            def spec(self):
                return ToolSpec(
                    name="blocked_tool",
                    description="test",
                    parameters={"type": "object", "properties": {}},
                    category=ToolCategory.MEMORY,
                )
            async def execute(self, arguments, context):
                return ToolResult.success_result("should not reach here")

        registry = ToolRegistry(db_pool)
        registry.register(DummyHandler())

        # Register a blocking hook
        async def blocker(ctx: HookContext) -> HookOutcome:
            return HookOutcome.blocked("hook says no")

        registry.hooks.register_function(HookEvent.BEFORE_TOOL_CALL, blocker, source="test")

        # Store tools config
        async with db_pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO config (key, value, description) VALUES ('tools', '{}'::jsonb, 'tools config') ON CONFLICT (key) DO NOTHING"
            )

        result = await registry.execute(
            "blocked_tool",
            {},
            ToolExecutionContext(tool_context=ToolContext.CHAT, call_id="test-1"),
        )
        assert result.success is False
        assert "hook says no" in (result.error or "")

    async def test_hook_mutates_arguments(self, db_pool):
        from core.tools.registry import ToolRegistry
        from core.tools.base import (
            ToolHandler, ToolSpec, ToolCategory, ToolContext,
            ToolExecutionContext, ToolResult,
        )
        from core.tools.hooks import HookEvent, HookContext, HookOutcome

        received_args = {}

        class CapturingHandler(ToolHandler):
            @property
            def spec(self):
                return ToolSpec(
                    name="capturing_tool",
                    description="test",
                    parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                    category=ToolCategory.MEMORY,
                )
            async def execute(self, arguments, context):
                received_args.update(arguments)
                return ToolResult.success_result("ok")

        registry = ToolRegistry(db_pool)
        registry.register(CapturingHandler())

        # Register a mutating hook
        async def mutator(ctx: HookContext) -> HookOutcome:
            args = dict(ctx.arguments or {})
            args["query"] = args.get("query", "") + " (enhanced by hook)"
            return HookOutcome.with_args(args)

        registry.hooks.register_function(HookEvent.BEFORE_TOOL_CALL, mutator, source="test")

        async with db_pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO config (key, value, description) VALUES ('tools', '{}'::jsonb, 'tools config') ON CONFLICT (key) DO NOTHING"
            )

        result = await registry.execute(
            "capturing_tool",
            {"query": "test"},
            ToolExecutionContext(tool_context=ToolContext.CHAT, call_id="test-2"),
        )
        assert result.success is True
        assert "enhanced by hook" in received_args.get("query", "")
