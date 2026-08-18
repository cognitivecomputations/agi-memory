"""Skill discovery tools.

The model-facing capability layer is skills; tools are implementation details.
These handlers let the model discover available skills and activate one during a
turn. `AgentLoop` watches successful `use_skill` calls and exposes that skill's
bound tools on the next iteration.
"""

from __future__ import annotations

import json
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from services.skill_runtime import get_skill_by_name, skill_bound_tools, skill_catalog
from skills.base import SkillCategory, SkillContext
from skills.loader import load_skills_from_dir

from .base import (
    ToolCategory,
    ToolContext,
    ToolErrorType,
    ToolExecutionContext,
    ToolHandler,
    ToolResult,
    ToolSpec,
)
from .self_extension import record_self_extension


AGENT_AUTHORED_SKILLS_DIR = Path.home() / ".hexis" / "skills" / "agent-authored"
_SKILL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")
_AGENT_SKILL_AUTHOR = "hexis"
_AGENT_SKILL_MANAGER = "author_skill"
_LEGACY_AGENT_PROVENANCE = "- Authored by Hexis via `author_skill`."


class ListSkillsHandler(ToolHandler):
    """List skills available in the current context."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="list_skills",
            description=(
                "List available skills. Use this when the task may need a capability "
                "that is not already active. Skills describe workflows and the tools "
                "they can unlock."
            ),
            parameters={"type": "object", "properties": {}},
            category=ToolCategory.EXTERNAL,
            energy_cost=0,
            is_read_only=True,
            allowed_contexts={ToolContext.CHAT, ToolContext.HEARTBEAT},
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        if not context.registry:
            return ToolResult.error_result("No registry available", ToolErrorType.EXECUTION_FAILED)
        skills = await skill_catalog(context.registry, context.tool_context)
        usable = sum(1 for s in skills if s.get("status") == "usable")
        return ToolResult.success_result(
            {
                "skills": skills,
                "acquirable": {
                    "propose_skill": (
                        "Create a pending, reviewable skill proposal when no "
                        "existing skill covers a reusable need. This does not "
                        "write a skill file."
                    ),
                    "author_skill": (
                        "Create a new skill with the author_skill tool — packaged "
                        "instructions plus the tools it binds."
                    ),
                    "mcp_skill": (
                        "New external integrations are added by installing a skill "
                        "manifest whose mcp block binds an MCP server; no core code "
                        "changes are needed."
                    ),
                },
            },
            f"{len(skills)} skill(s): {usable} usable",
        )


class UseSkillHandler(ToolHandler):
    """Activate a skill and return its instructions."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="use_skill",
            description=(
                "Activate a named skill for this turn. Returns the skill instructions "
                "and unlocks its bound tools for subsequent tool calls."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The skill name from list_skills, e.g. research or meeting-prep.",
                    },
                },
                "required": ["name"],
            },
            category=ToolCategory.EXTERNAL,
            energy_cost=0,
            is_read_only=True,
            allowed_contexts={ToolContext.CHAT, ToolContext.HEARTBEAT},
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        if not context.registry:
            return ToolResult.error_result("No registry available", ToolErrorType.EXECUTION_FAILED)
        name = str(arguments.get("name") or "").strip()
        if not name:
            return ToolResult.error_result("Skill name is required", ToolErrorType.INVALID_PARAMS)
        skill = await get_skill_by_name(context.registry, context.tool_context, name)
        if not skill:
            return ToolResult.error_result(f"Unknown or unavailable skill: {name}", ToolErrorType.INVALID_PARAMS)

        native_bound = [
            t for t in skill_bound_tools(skill)
            if not t.startswith("mcp_") and context.registry.get_spec(t) is not None
        ]
        payload: dict[str, Any] = {
            "name": skill.name,
            "description": skill.description,
            "instructions": skill.content,
            "bound_tools": native_bound,
        }

        if skill.mcp_binding is not None:
            activation = await self._activate_mcp(skill, context)
            payload.update(activation)
            if activation.get("status") != "activated":
                # No dead-end: instructions and the exact next step are still
                # delivered; only the MCP tools stay locked.
                return ToolResult.success_result(
                    payload,
                    f"Skill {skill.name}: {activation.get('status')} — {activation.get('next_step', '')}".strip(),
                )
            payload["bound_tools"] = [*native_bound, *activation.get("mcp_tools", [])]

        return ToolResult.success_result(payload, f"Activated skill: {skill.name}")

    async def _activate_mcp(
        self, skill, context: ToolExecutionContext
    ) -> dict[str, Any]:
        """Lazily connect the skill's MCP server and register ONLY its
        manifest-bound tools (#41). Returns a status dict for the payload."""
        import os

        from core.tools.mcp_runtime import MCPRuntime
        from core.tools.config import MCPServerConfig

        binding = skill.mcp_binding
        missing_env = [v for v in binding.env_requires if not os.environ.get(v)]
        if missing_env:
            return {
                "status": "needs_setup",
                "missing": [f"missing env var: {v}" for v in missing_env],
                "next_step": (
                    f"Set {', '.join(missing_env)} in the service environment and "
                    "call use_skill again."
                ),
            }

        if binding.command:
            server_config = MCPServerConfig(
                name=binding.server, command=binding.command, args=list(binding.args)
            )
        else:
            config = await context.registry.get_config()
            server_config = next(
                (c for c in (config.mcp_servers or []) if c.name == binding.server and c.enabled),
                None,
            )
            if server_config is None:
                return {
                    "status": "unavailable",
                    "next_step": (
                        f"Add an MCP server named '{binding.server}' to the tools config "
                        "(mcp_servers), or add a command to the skill manifest."
                    ),
                }

        runtime = MCPRuntime.instance()
        result = await runtime.ensure_connected(server_config)
        if not result.get("connected"):
            return {
                "status": "connection_failed",
                "error": result.get("error"),
                "next_step": result.get("next_step"),
            }
        mcp_tools = runtime.register_into(
            context.registry, binding.server, skill_bound_tools(skill)
        )
        if not mcp_tools:
            return {
                "status": "connection_failed",
                "error": (
                    "server connected but exposes none of the tools this skill binds "
                    f"({', '.join(t for t in skill_bound_tools(skill) if t.startswith('mcp_'))})"
                ),
                "next_step": "Check the skill manifest's bound_tools against the server's tool list.",
            }
        return {"status": "activated", "mcp_tools": mcp_tools}


class ProposeSkillHandler(ToolHandler):
    """Create a durable, reviewable skill proposal without writing files."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="propose_skill",
            description=(
                "Draft a reusable Hexis skill proposal when no existing skill covers "
                "a capability gap or the user asks for a new reusable behavior. "
                "This only creates a pending review item; it does not write or activate "
                "a skill file."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "need": {
                        "type": "string",
                        "description": "The reusable capability gap or user-requested behavior this skill would cover.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Skill name, lowercase kebab-case, e.g. invoice-triage.",
                    },
                    "description": {
                        "type": "string",
                        "description": "One concise sentence describing when to use the skill.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Markdown skill instructions, including method and quality guidance.",
                    },
                    "category": {
                        "type": "string",
                        "enum": [c.value for c in SkillCategory],
                        "description": "Skill category.",
                        "default": SkillCategory.OTHER.value,
                    },
                    "contexts": {
                        "type": "array",
                        "items": {"type": "string", "enum": [c.value for c in SkillContext]},
                        "description": "Contexts where the skill can activate.",
                        "default": [SkillContext.CHAT.value, SkillContext.HEARTBEAT.value],
                    },
                    "bound_tools": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Existing tools this skill may unlock.",
                        "default": [],
                    },
                    "requires_tools": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Existing tools required for this skill to load.",
                        "default": [],
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["create", "update"],
                        "description": "create refuses to shadow an existing skill; update requires Hexis-managed ownership.",
                        "default": "create",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Why this reusable skill is worth reviewing.",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Confidence that this should become durable. Defaults to 0.75.",
                    },
                    "source_memory_ids": {
                        "type": "array",
                        "items": {"type": "string", "format": "uuid"},
                        "description": "Optional source memories supporting this proposal.",
                    },
                    "source_unit_ids": {
                        "type": "array",
                        "items": {"type": "string", "format": "uuid"},
                        "description": "Optional raw experience units supporting this proposal.",
                    },
                },
                "required": ["need", "name", "description", "content", "rationale"],
            },
            category=ToolCategory.EXTERNAL,
            energy_cost=1,
            is_read_only=False,
            requires_approval=False,
            supports_parallel=False,
            allowed_contexts={ToolContext.CHAT, ToolContext.HEARTBEAT},
        )

    def validate(self, arguments: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        name = str(arguments.get("name") or "").strip()
        if not _SKILL_NAME_RE.match(name):
            errors.append("name must be lowercase kebab-case/underscore, 2-64 chars")
        if not str(arguments.get("need") or "").strip():
            errors.append("need is required")
        if not str(arguments.get("description") or "").strip():
            errors.append("description is required")
        if not str(arguments.get("rationale") or "").strip():
            errors.append("rationale is required")
        content = str(arguments.get("content") or "").strip()
        if len(content) < 120:
            errors.append("content must be substantive (at least 120 characters)")
        mode = str(arguments.get("mode") or "create")
        if mode not in {"create", "update"}:
            errors.append("mode must be create or update")
        for ctx in arguments.get("contexts") or [SkillContext.CHAT.value, SkillContext.HEARTBEAT.value]:
            if str(ctx) not in {c.value for c in SkillContext}:
                errors.append(f"unknown context: {ctx}")
        category = str(arguments.get("category") or SkillCategory.OTHER.value)
        if category not in {c.value for c in SkillCategory}:
            errors.append(f"unknown category: {category}")
        for field in ("source_memory_ids", "source_unit_ids"):
            for value in arguments.get(field) or []:
                try:
                    uuid.UUID(str(value))
                except ValueError:
                    errors.append(f"{field} contains a non-UUID value: {value}")
        confidence = arguments.get("confidence")
        if confidence is not None:
            try:
                numeric_confidence = float(confidence)
                if not 0 <= numeric_confidence <= 1:
                    errors.append("confidence must be between 0 and 1")
            except (TypeError, ValueError):
                errors.append("confidence must be numeric")
        return errors

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        if not context.registry:
            return ToolResult.error_result("Skill proposals require a registry reference", ToolErrorType.EXECUTION_FAILED)

        name = str(arguments["name"]).strip()
        mode = str(arguments.get("mode") or "create")
        existing = await get_skill_by_name(context.registry, context.tool_context, name)
        try:
            _skill_dir, skill_path = _agent_skill_path(name)
        except ValueError as exc:
            return ToolResult.error_result(str(exc), ToolErrorType.PATH_NOT_ALLOWED)

        if mode == "create":
            if existing or skill_path.exists():
                return ToolResult.error_result(
                    f"Skill '{name}' already exists; propose mode='update' for a Hexis-managed skill or choose a new name.",
                    ToolErrorType.INVALID_PARAMS,
                )
        else:
            if not existing or not skill_path.exists():
                return ToolResult.error_result(
                    f"Skill '{name}' does not exist as a Hexis-managed agent-authored skill; propose mode='create' with a new name.",
                    ToolErrorType.FILE_NOT_FOUND,
                )
            managed, detail, _provenance = _agent_skill_ownership(skill_path)
            if not managed:
                return ToolResult.error_result(detail, ToolErrorType.PERMISSION_DENIED)

        bound_tools = _string_list(arguments.get("bound_tools"))
        requires_tools = _string_list(arguments.get("requires_tools"))
        if not requires_tools:
            requires_tools = bound_tools[:]
        known_tools = set(context.registry.list_names())
        unknown = sorted((set(bound_tools) | set(requires_tools)) - known_tools)
        if unknown:
            return ToolResult.error_result(
                "Unknown tool(s) in skill proposal metadata: " + ", ".join(unknown),
                ToolErrorType.INVALID_PARAMS,
            )

        proposal = {
            "name": name,
            "description": str(arguments["description"]).strip(),
            "content": str(arguments["content"]).strip(),
            "category": str(arguments.get("category") or SkillCategory.OTHER.value),
            "contexts": [str(v) for v in (arguments.get("contexts") or [SkillContext.CHAT.value, SkillContext.HEARTBEAT.value])],
            "bound_tools": bound_tools,
            "requires_tools": requires_tools,
            "mode": mode,
            "rationale": str(arguments["rationale"]).strip(),
            "confidence": float(arguments.get("confidence", 0.75)),
        }
        evidence = {
            "origin": "on_demand",
            "kind": "skill_proposal",
            "need": str(arguments["need"]).strip(),
            "tool_context": context.tool_context.value,
            "call_id": context.call_id,
            "session_id": context.session_id,
            "heartbeat_id": context.heartbeat_id,
            "source_memory_ids": _string_list(arguments.get("source_memory_ids")),
            "source_unit_ids": _string_list(arguments.get("source_unit_ids")),
        }

        async with context.registry.pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT create_on_demand_skill_proposal($1::jsonb, $2::jsonb)",
                json.dumps(proposal),
                json.dumps(evidence),
            )
        created = json.loads(raw) if isinstance(raw, str) else raw
        proposal_id = str(created.get("proposal_id")) if created else None
        return ToolResult.success_result(
            {
                **(created or {}),
                "proposal_id": proposal_id,
                "name": name,
                "mode": mode,
                "status": "pending",
                "writes_skill_file": False,
                "next_step": (
                    f"Review with `hexis skills review {proposal_id} --action apply` "
                    "or call review_skill_proposal with explicit approval."
                ) if proposal_id else "Run `hexis skills proposals` to inspect pending proposals.",
            },
            f"Proposed skill '{name}' for review",
        )


class AuthorSkillHandler(ToolHandler):
    """Create or update an explicitly agent-authored skill document."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="author_skill",
            description=(
                "Create or update a Hexis skill. Use this when a useful workflow "
                "should become reusable future behavior. Writes only to the "
                "agent-authored skill directory, verifies ownership before updates, "
                "and validates the resulting SKILL.md before saving."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name, lowercase kebab-case, e.g. weekly-review.",
                    },
                    "description": {
                        "type": "string",
                        "description": "One concise sentence describing when to use the skill.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Markdown skill instructions, including method and quality guidance.",
                    },
                    "category": {
                        "type": "string",
                        "enum": [c.value for c in SkillCategory],
                        "description": "Skill category.",
                        "default": SkillCategory.OTHER.value,
                    },
                    "contexts": {
                        "type": "array",
                        "items": {"type": "string", "enum": [c.value for c in SkillContext]},
                        "description": "Contexts where the skill can activate.",
                        "default": [SkillContext.CHAT.value, SkillContext.HEARTBEAT.value],
                    },
                    "bound_tools": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Existing tools this skill may unlock.",
                        "default": [],
                    },
                    "requires_tools": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Existing tools required for this skill to load.",
                        "default": [],
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["create", "update"],
                        "description": "create refuses to overwrite; update requires an existing skill.",
                        "default": "create",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Why this skill is worth keeping.",
                    },
                    "proposal_id": {
                        "type": "string",
                        "format": "uuid",
                        "description": "Optional durable improvement proposal that supports this skill.",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Optional confidence of the supporting improvement review.",
                    },
                    "source_memory_ids": {
                        "type": "array",
                        "items": {"type": "string", "format": "uuid"},
                        "description": "Optional source memories supporting this skill.",
                    },
                    "source_unit_ids": {
                        "type": "array",
                        "items": {"type": "string", "format": "uuid"},
                        "description": "Optional raw experience units supporting this skill.",
                    },
                    "evidence_digest": {
                        "type": "string",
                        "description": "Optional digest of the reviewed evidence window.",
                    },
                },
                "required": ["name", "description", "content"],
            },
            category=ToolCategory.EXTERNAL,
            energy_cost=3,
            is_read_only=False,
            requires_approval=True,
            supports_parallel=False,
            allowed_contexts={ToolContext.CHAT, ToolContext.HEARTBEAT},
        )

    def validate(self, arguments: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        name = str(arguments.get("name") or "").strip()
        if not _SKILL_NAME_RE.match(name):
            errors.append("name must be lowercase kebab-case/underscore, 2-64 chars")
        if not str(arguments.get("description") or "").strip():
            errors.append("description is required")
        content = str(arguments.get("content") or "").strip()
        if len(content) < 120:
            errors.append("content must be substantive (at least 120 characters)")
        mode = str(arguments.get("mode") or "create")
        if mode not in {"create", "update"}:
            errors.append("mode must be create or update")
        for ctx in arguments.get("contexts") or [SkillContext.CHAT.value, SkillContext.HEARTBEAT.value]:
            if str(ctx) not in {c.value for c in SkillContext}:
                errors.append(f"unknown context: {ctx}")
        category = str(arguments.get("category") or SkillCategory.OTHER.value)
        if category not in {c.value for c in SkillCategory}:
            errors.append(f"unknown category: {category}")
        for field in ("proposal_id",):
            value = arguments.get(field)
            if value:
                try:
                    uuid.UUID(str(value))
                except ValueError:
                    errors.append(f"{field} must be a UUID")
        for field in ("source_memory_ids", "source_unit_ids"):
            for value in arguments.get(field) or []:
                try:
                    uuid.UUID(str(value))
                except ValueError:
                    errors.append(f"{field} contains a non-UUID value: {value}")
        confidence = arguments.get("confidence")
        if confidence is not None:
            try:
                numeric_confidence = float(confidence)
                if not 0 <= numeric_confidence <= 1:
                    errors.append("confidence must be between 0 and 1")
            except (TypeError, ValueError):
                errors.append("confidence must be numeric")
        return errors

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        if not context.registry:
            return ToolResult.error_result("Skill authoring requires a registry reference", ToolErrorType.EXECUTION_FAILED)

        name = str(arguments["name"]).strip()
        description = str(arguments["description"]).strip()
        content = str(arguments["content"]).strip()
        category = str(arguments.get("category") or SkillCategory.OTHER.value)
        contexts = [str(v) for v in (arguments.get("contexts") or [SkillContext.CHAT.value, SkillContext.HEARTBEAT.value])]
        bound_tools = _string_list(arguments.get("bound_tools"))
        requires_tools = _string_list(arguments.get("requires_tools"))
        if not requires_tools:
            requires_tools = bound_tools[:]

        known_tools = set(context.registry.list_names())
        unknown = sorted((set(bound_tools) | set(requires_tools)) - known_tools)
        if unknown:
            return ToolResult.error_result(
                "Unknown tool(s) in skill metadata: " + ", ".join(unknown),
                ToolErrorType.INVALID_PARAMS,
            )

        mode = str(arguments.get("mode") or "create")
        try:
            skill_dir, path = _agent_skill_path(name)
        except ValueError as exc:
            return ToolResult.error_result(str(exc), ToolErrorType.PATH_NOT_ALLOWED)
        if mode == "create" and path.exists():
            return ToolResult.error_result(
                f"Skill '{name}' already exists; use mode='update' to replace it.",
                ToolErrorType.INVALID_PARAMS,
            )
        if mode == "update" and not path.exists():
            return ToolResult.error_result(
                f"Skill '{name}' does not exist; use mode='create' to create it.",
                ToolErrorType.FILE_NOT_FOUND,
            )

        existing_provenance: dict[str, Any] = {}
        if mode == "update":
            managed, detail, existing_provenance = _agent_skill_ownership(path)
            if not managed:
                return ToolResult.error_result(detail, ToolErrorType.PERMISSION_DENIED)

        evidence_provenance = {
            "proposal_id": arguments.get("proposal_id", existing_provenance.get("proposal_id")),
            "confidence": arguments.get("confidence", existing_provenance.get("confidence")),
            "source_memory_ids": _string_list(
                arguments.get("source_memory_ids", existing_provenance.get("source_memory_ids"))
            ),
            "source_unit_ids": _string_list(
                arguments.get("source_unit_ids", existing_provenance.get("source_unit_ids"))
            ),
            "evidence_digest": arguments.get(
                "evidence_digest", existing_provenance.get("evidence_digest")
            ),
        }

        skill_text = _render_skill_markdown(
            name=name,
            description=description,
            category=category,
            contexts=contexts,
            bound_tools=bound_tools,
            requires_tools=requires_tools,
            content=content,
            rationale=str(arguments.get("rationale") or "").strip(),
            created_at=existing_provenance.get("created_at"),
            evidence_provenance=evidence_provenance,
        )
        valid, detail = _validate_skill_text(name, skill_text)
        if not valid:
            return ToolResult.error_result(detail, ToolErrorType.INVALID_PARAMS)

        skill_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(skill_text, encoding="utf-8")

        # Substrate-change visibility (#93): journal + web-inbox notice.
        await record_self_extension(
            context.registry.pool,
            summary=f"Agent {mode}d skill '{name}'",
            notice=(
                f"I {'revised' if mode == 'update' else 'wrote'} a skill for myself: "
                f"'{name}' — {description}"
            ),
            detail={
                "skill": name,
                "mode": mode,
                "path": str(path),
                "bound_tools": bound_tools,
            },
        )

        return ToolResult.success_result(
            {
                "skill": name,
                "path": str(path),
                "mode": mode,
                "contexts": contexts,
                "bound_tools": bound_tools,
                "requires_tools": requires_tools,
                "provenance": {
                    "authored_by": _AGENT_SKILL_AUTHOR,
                    "managed_by": _AGENT_SKILL_MANAGER,
                    **{key: value for key, value in evidence_provenance.items() if value not in (None, [], "")},
                },
                "discoverable_next_turn": True,
            },
            f"{mode.title()}d skill '{name}'",
        )


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def _agent_skill_path(name: str) -> tuple[Path, Path]:
    root = AGENT_AUTHORED_SKILLS_DIR.expanduser()
    skill_dir = root / name
    path = skill_dir / "SKILL.md"
    if skill_dir.is_symlink() or path.is_symlink():
        raise ValueError(
            f"Refusing skill '{name}': agent-authored skill paths may not be symlinks. "
            "Use a regular directory under the configured agent-authored skill root."
        )
    resolved_root = root.resolve(strict=False)
    resolved_skill_dir = skill_dir.resolve(strict=False)
    if not resolved_skill_dir.is_relative_to(resolved_root):
        raise ValueError(
            f"Refusing skill '{name}': target escapes the agent-authored skill root."
        )
    return skill_dir, path


def _agent_skill_ownership(path: Path) -> tuple[bool, str, dict[str, Any]]:
    parsed = [
        skill
        for skill in load_skills_from_dir(path.parent)
        if Path(skill.source).resolve(strict=False) == path.resolve(strict=False)
    ]
    if len(parsed) == 1:
        provenance = parsed[0].provenance
        if (
            provenance.get("authored_by") == _AGENT_SKILL_AUTHOR
            and provenance.get("managed_by") == _AGENT_SKILL_MANAGER
        ):
            return True, "structured agent ownership verified", dict(provenance)

    try:
        existing_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return False, f"Could not verify ownership of {path}: {exc}", {}
    if _LEGACY_AGENT_PROVENANCE in existing_text:
        return True, "legacy agent ownership verified", {}
    return (
        False,
        f"Refusing to replace '{path}': it is not marked as managed by Hexis "
        "author_skill. Choose a new skill name or edit the user-authored file manually.",
        {},
    )


def _render_skill_markdown(
    *,
    name: str,
    description: str,
    category: str,
    contexts: list[str],
    bound_tools: list[str],
    requires_tools: list[str],
    content: str,
    rationale: str,
    created_at: str | None = None,
    evidence_provenance: dict[str, Any] | None = None,
) -> str:
    now = datetime.now(timezone.utc).isoformat()
    metadata = [
        "---",
        f"name: {json.dumps(name)}",
        f"description: {json.dumps(description)}",
        f"category: {json.dumps(category)}",
        "requires:",
        f"  tools: {json.dumps(requires_tools)}",
        f"contexts: {json.dumps(contexts)}",
        f"bound_tools: {json.dumps(bound_tools)}",
        "provenance:",
        f"  authored_by: {json.dumps(_AGENT_SKILL_AUTHOR)}",
        f"  managed_by: {json.dumps(_AGENT_SKILL_MANAGER)}",
        f"  created_at: {json.dumps(created_at or now)}",
        f"  updated_at: {json.dumps(now)}",
    ]
    for key in ("proposal_id", "confidence", "source_memory_ids", "source_unit_ids", "evidence_digest"):
        value = (evidence_provenance or {}).get(key)
        if value not in (None, [], ""):
            metadata.append(f"  {key}: {json.dumps(value)}")
    metadata.extend(["---", ""])
    body = content.rstrip() + "\n"
    footer = [
        "",
        "## Provenance",
        "",
        "- Authored by Hexis via `author_skill`.",
        f"- Updated at: {now}",
    ]
    if rationale:
        footer.append(f"- Rationale: {rationale}")
    return "\n".join(metadata) + body + "\n".join(footer) + "\n"


def _validate_skill_text(expected_name: str, skill_text: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        skill_dir = Path(tmp) / expected_name
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(skill_text, encoding="utf-8")
        parsed = load_skills_from_dir(Path(tmp))
    if len(parsed) != 1:
        return False, "Generated skill did not parse as exactly one SKILL.md"
    skill = parsed[0]
    if skill.name != expected_name:
        return False, f"Generated skill parsed with unexpected name: {skill.name}"
    if not skill.description.strip():
        return False, "Generated skill has no description"
    if len(skill.content) < 120:
        return False, "Generated skill content is too short"
    return True, "ok"


def _proposal_output(row: Any) -> dict[str, Any]:
    output = dict(row)
    for field in ("id",):
        if output.get(field) is not None:
            output[field] = str(output[field])
    for field in ("source_memory_ids", "source_unit_ids"):
        output[field] = [str(value) for value in output.get(field) or []]
    for field in ("created_at", "updated_at", "reviewed_at", "applied_at"):
        value = output.get(field)
        if value is not None:
            output[field] = value.isoformat() if hasattr(value, "isoformat") else str(value)
    return output


class ListSkillProposalsHandler(ToolHandler):
    """List durable self-improvement proposals awaiting explicit review."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="list_skill_proposals",
            description=(
                "List background skill-improvement proposals and their evidence lineage. "
                "No proposal changes behavior until review_skill_proposal applies it."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["pending", "applied", "rejected", "all"],
                        "default": "pending",
                    },
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
                },
            },
            category=ToolCategory.EXTERNAL,
            energy_cost=0,
            is_read_only=True,
            allowed_contexts={ToolContext.CHAT, ToolContext.HEARTBEAT},
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        if not context.registry:
            return ToolResult.error_result("Proposal review requires a registry reference", ToolErrorType.EXECUTION_FAILED)
        status = str(arguments.get("status") or "pending")
        limit = min(max(int(arguments.get("limit") or 20), 1), 100)
        if status not in {"pending", "applied", "rejected", "all"}:
            return ToolResult.error_result("status must be pending, applied, rejected, or all", ToolErrorType.INVALID_PARAMS)
        async with context.registry.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, status, name, description, category, contexts,
                       bound_tools, requires_tools, mode, rationale, confidence,
                       source_memory_ids, source_unit_ids, evidence_digest,
                       last_error, created_at, reviewed_at, applied_at
                FROM skill_improvement_proposals
                WHERE ($1::text = 'all' OR status = $1::text)
                ORDER BY created_at DESC, id
                LIMIT $2::int
                """,
                status,
                limit,
            )
        proposals = [_proposal_output(row) for row in rows]
        return ToolResult.success_result(
            {"proposals": proposals}, f"{len(proposals)} skill proposal(s)"
        )


class ReviewSkillProposalHandler(ToolHandler):
    """Apply, reject, or reopen one durable proposal with explicit approval."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="review_skill_proposal",
            description=(
                "Explicitly apply, reject, or reopen a skill-improvement proposal. "
                "Apply writes through author_skill ownership validation; reject keeps the proposal recoverable."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "proposal_id": {"type": "string", "format": "uuid"},
                    "action": {"type": "string", "enum": ["apply", "reject", "reopen"]},
                },
                "required": ["proposal_id", "action"],
            },
            category=ToolCategory.EXTERNAL,
            energy_cost=3,
            is_read_only=False,
            requires_approval=True,
            supports_parallel=False,
            allowed_contexts={ToolContext.CHAT, ToolContext.HEARTBEAT},
        )

    def validate(self, arguments: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        try:
            uuid.UUID(str(arguments.get("proposal_id") or ""))
        except ValueError:
            errors.append("proposal_id must be a UUID")
        if arguments.get("action") not in {"apply", "reject", "reopen"}:
            errors.append("action must be apply, reject, or reopen")
        return errors

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        if not context.registry:
            return ToolResult.error_result("Proposal review requires a registry reference", ToolErrorType.EXECUTION_FAILED)
        proposal_id = str(arguments["proposal_id"])
        action = str(arguments["action"])
        async with context.registry.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM skill_improvement_proposals WHERE id = $1::uuid",
                proposal_id,
            )
            if not row:
                return ToolResult.error_result(f"Skill proposal not found: {proposal_id}", ToolErrorType.FILE_NOT_FOUND)
            if action != "apply":
                try:
                    transitioned = await conn.fetchval(
                        "SELECT transition_skill_improvement_proposal($1::uuid, $2::text)",
                        proposal_id,
                        action,
                    )
                except Exception as exc:
                    return ToolResult.error_result(str(exc), ToolErrorType.INVALID_PARAMS)
                return ToolResult.success_result(
                    {"proposal_id": proposal_id, "action": action, "proposal": transitioned},
                    f"{action.title()}ed skill proposal '{row['name']}'",
                )

        try:
            _skill_dir, skill_path = _agent_skill_path(str(row["name"]))
        except ValueError as exc:
            return ToolResult.error_result(str(exc), ToolErrorType.PATH_NOT_ALLOWED)
        if skill_path.exists():
            parsed = load_skills_from_dir(skill_path.parent)
            if len(parsed) == 1 and parsed[0].provenance.get("proposal_id") == proposal_id:
                author_result = ToolResult.success_result({"skill": row["name"], "already_written": True})
            else:
                author_result = await self._apply_with_author(row, proposal_id, context)
        else:
            author_result = await self._apply_with_author(row, proposal_id, context)

        if not author_result.success:
            async with context.registry.pool.acquire() as conn:
                try:
                    await conn.fetchval(
                        "SELECT transition_skill_improvement_proposal($1::uuid, 'error', $2::text)",
                        proposal_id,
                        author_result.error,
                    )
                except Exception:
                    pass
            return author_result

        async with context.registry.pool.acquire() as conn:
            try:
                transitioned = await conn.fetchval(
                    "SELECT transition_skill_improvement_proposal($1::uuid, 'apply')",
                    proposal_id,
                )
            except Exception as exc:
                return ToolResult.error_result(
                    f"Skill file was written but proposal state could not be finalized: {exc}. "
                    "Retry this same apply action; proposal provenance makes it idempotent.",
                    ToolErrorType.EXECUTION_FAILED,
                )
        return ToolResult.success_result(
            {
                "proposal_id": proposal_id,
                "action": "apply",
                "skill": author_result.output,
                "proposal": transitioned,
            },
            f"Applied skill proposal '{row['name']}'",
        )

    @staticmethod
    async def _apply_with_author(row: Any, proposal_id: str, context: ToolExecutionContext) -> ToolResult:
        arguments = {
                "name": row["name"],
                "description": row["description"],
                "content": row["content"],
                "category": row["category"],
                "contexts": list(row["contexts"]),
                "bound_tools": list(row["bound_tools"]),
                "requires_tools": list(row["requires_tools"]),
                "mode": row["mode"],
                "rationale": row["rationale"],
                "proposal_id": proposal_id,
                "confidence": row["confidence"],
                "source_memory_ids": [str(value) for value in row["source_memory_ids"]],
                "source_unit_ids": [str(value) for value in row["source_unit_ids"]],
                "evidence_digest": row["evidence_digest"],
        }
        author = AuthorSkillHandler()
        errors = author.validate(arguments)
        if errors:
            return ToolResult.error_result("; ".join(errors), ToolErrorType.INVALID_PARAMS)
        return await author.execute(arguments, context)


def create_skill_tools() -> list[ToolHandler]:
    return [
        ListSkillsHandler(),
        UseSkillHandler(),
        ProposeSkillHandler(),
        AuthorSkillHandler(),
        ListSkillProposalsHandler(),
        ReviewSkillProposalHandler(),
    ]
