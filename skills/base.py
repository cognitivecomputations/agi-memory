"""
Hexis Skills System - Base Types

SkillSpec: the parsed representation of a skill markdown file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SkillContext(str, Enum):
    """Contexts in which a skill can be active."""

    HEARTBEAT = "heartbeat"
    CHAT = "chat"
    MCP = "mcp"


class SkillCategory(str, Enum):
    """Skill categories for organization and discovery."""

    RESEARCH = "research"
    PRODUCTIVITY = "productivity"
    COMMUNICATION = "communication"
    KNOWLEDGE = "knowledge"
    ANALYTICS = "analytics"
    CREATIVE = "creative"
    SYSTEM = "system"
    OTHER = "other"


@dataclass
class InstallMethod:
    """A method for installing a skill's binary dependency."""

    kind: str  # brew, apt, pip, npm, etc.
    package: str  # package name (formula for brew, package for apt/pip)
    bins: list[str] = field(default_factory=list)  # binaries provided

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InstallMethod":
        return cls(
            kind=str(d.get("kind", "")),
            package=str(d.get("package", d.get("formula", ""))),
            bins=list(d.get("bins", [])),
        )


@dataclass
class MCPBinding:
    """A skill's binding to an MCP server (#41): the skill is the model-facing
    capability; the server is a transport detail connected lazily at
    activation. Credentials stay env-var NAMES only — values never leave the
    process environment."""

    server: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env_requires: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MCPBinding | None":
        server = str(d.get("server", "")).strip()
        if not server:
            return None
        raw_args = d.get("args", [])
        if isinstance(raw_args, str):
            raw_args = [raw_args]
        raw_env = d.get("env_requires", [])
        if isinstance(raw_env, str):
            raw_env = [raw_env]
        command = d.get("command")
        return cls(
            server=server,
            command=str(command) if command else None,
            args=[str(a) for a in raw_args],
            env_requires=[str(v) for v in raw_env],
        )


@dataclass
class SkillSpec:
    """Parsed representation of a skill document."""

    name: str
    description: str
    content: str  # Markdown body (after frontmatter)
    requires_tools: list[str] = field(default_factory=list)
    requires_config: list[str] = field(default_factory=list)
    requires_bins: list[str] = field(default_factory=list)  # J.1: binary deps
    requires_env: list[str] = field(default_factory=list)  # J.1: env var deps
    install_methods: list[InstallMethod] = field(default_factory=list)  # J.1
    category: SkillCategory = SkillCategory.OTHER  # J.1
    os_support: list[str] = field(default_factory=lambda: ["darwin", "linux"])  # J.1
    bound_tools: list[str] = field(default_factory=list)  # J.4: tool names to bind
    # Words a person would actually use for this skill. Selection scores literal
    # token overlap against the skill's *name*, so `gmail-actions` scored zero on
    # "email" and `calendar` scored zero on "book time with Sarah". Aliases are
    # scored exactly like name tokens — they are the user's vocabulary, not ours.
    aliases: list[str] = field(default_factory=list)
    # Exact, manifest-owned phrases that activate a capability even when the
    # semantic score distribution is flat or misleading. Use sparingly for
    # unambiguous user language (for example "screenshot"), not broad topics.
    activation_phrases: list[str] = field(default_factory=list)
    contexts: list[SkillContext] = field(
        default_factory=lambda: [SkillContext.HEARTBEAT, SkillContext.CHAT]
    )
    source: str = ""  # File path or plugin ID that provided this skill
    enabled: bool = True  # Can be disabled via config
    provenance: dict[str, Any] = field(default_factory=dict)
    mcp_binding: MCPBinding | None = None  # #41: MCP server this skill binds
    # Optional consent-first automation proposal. Loading/installing the skill
    # may register this block for review, but never schedules it directly.
    blueprint: dict[str, Any] | None = None

    def requirements_met(
        self,
        available_tools: set[str],
        available_config: set[str] | None = None,
    ) -> bool:
        """Check if all requirements are satisfied. MCP-bound tools (mcp_*)
        exist only after activation and are never required up front."""
        for tool in self.requires_tools:
            if tool.startswith("mcp_"):
                continue
            if tool not in available_tools:
                return False
        if available_config is not None:
            for key in self.requires_config:
                if key not in available_config:
                    return False
        return True

    def check_bins_available(self) -> list[str]:
        """Return list of missing binary dependencies."""
        import shutil

        missing = []
        for b in self.requires_bins:
            if not shutil.which(b):
                missing.append(b)
        return missing

    def check_env_available(self) -> list[str]:
        """Return list of missing environment variables."""
        import os

        missing = []
        for var in self.requires_env:
            if not os.environ.get(var):
                missing.append(var)
        return missing

    def check_os_support(self) -> bool:
        """Check if the current OS is supported."""
        import sys

        return sys.platform in self.os_support

    def full_requirements_met(
        self,
        available_tools: set[str],
        available_config: set[str] | None = None,
    ) -> tuple[bool, list[str]]:
        """
        Full requirements check including bins, env vars, and OS.

        Returns (met, reasons) where reasons lists what's missing.
        """
        reasons: list[str] = []

        if not self.enabled:
            reasons.append("skill disabled")

        if not self.check_os_support():
            import sys

            reasons.append(f"OS {sys.platform} not in {self.os_support}")

        for tool in self.requires_tools:
            if tool.startswith("mcp_"):
                continue
            if tool not in available_tools:
                reasons.append(f"missing tool: {tool}")

        if available_config is not None:
            for key in self.requires_config:
                if key not in available_config:
                    reasons.append(f"missing config: {key}")

        missing_bins = self.check_bins_available()
        for b in missing_bins:
            reasons.append(f"missing binary: {b}")

        missing_env = self.check_env_available()
        for v in missing_env:
            reasons.append(f"missing env var: {v}")

        return (len(reasons) == 0, reasons)

    def usability(
        self,
        available_tools: set[str],
        config_server_names: set[str] | None = None,
    ) -> tuple[str, list[str], str | None]:
        """Tri-state capability status (#39): ``usable`` | ``needs_setup`` |
        ``unavailable``, with what's missing and the exact next step — a
        capability question must never dead-end in a bare "no".

        MCP-bound tools (``mcp_*``) are excluded from the native-tool check:
        they exist only after activation, by design.
        """
        missing: list[str] = []
        import sys

        if not self.enabled:
            return (
                "unavailable",
                ["skill disabled"],
                "Enable the skill in config to use it.",
            )
        if not self.check_os_support():
            return (
                "unavailable",
                [f"OS {sys.platform} not in {self.os_support}"],
                None,
            )

        missing_tools = [
            t
            for t in self.requires_tools
            if not t.startswith("mcp_") and t not in available_tools
        ]
        if missing_tools:
            return (
                "unavailable",
                [f"missing tool: {t}" for t in missing_tools],
                "These native tools are not registered in this runtime; enable them in tools config.",
            )

        if self.mcp_binding and self.mcp_binding.command is None:
            known = config_server_names or set()
            if self.mcp_binding.server not in known:
                return (
                    "unavailable",
                    [f"mcp server not configured: {self.mcp_binding.server}"],
                    (
                        f"Add an MCP server named '{self.mcp_binding.server}' to the tools "
                        "config (mcp_servers), or add a command to the skill manifest."
                    ),
                )

        env_wanted = list(
            dict.fromkeys(
                [
                    *self.requires_env,
                    *(self.mcp_binding.env_requires if self.mcp_binding else []),
                ]
            )
        )
        import os

        missing_env = [v for v in env_wanted if not os.environ.get(v)]
        if missing_env:
            return (
                "needs_setup",
                [f"missing env var: {v}" for v in missing_env],
                f"Set {', '.join(missing_env)} in the service environment and restart.",
            )

        missing_bins = self.check_bins_available()
        if missing_bins:
            steps = [
                f"{m.kind} install {m.package}"
                for m in self.install_methods
                if set(m.bins) & set(missing_bins) or not m.bins
            ]
            return (
                "needs_setup",
                [f"missing binary: {b}" for b in missing_bins],
                f"Install with: {steps[0]}"
                if steps
                else f"Install {', '.join(missing_bins)} and retry.",
            )

        return ("usable", missing, None)

    def to_prompt_block(self) -> str:
        """Format this skill's full instructions (returned by `use_skill`)."""
        return f'<skill name="{self.name}">\n{self.content}\n</skill>'

    def to_index_line(self) -> str:
        """One-line entry for the compact skill index in the system prompt."""
        desc = " ".join(self.description.split())
        return f"- {self.name}: {desc}" if desc else f"- {self.name}"

    @classmethod
    def from_frontmatter(
        cls, metadata: dict[str, Any], content: str, source: str = ""
    ) -> "SkillSpec":
        """Create a SkillSpec from parsed YAML frontmatter and markdown body."""
        # Parse contexts
        raw_contexts = metadata.get("contexts", ["heartbeat", "chat"])
        if isinstance(raw_contexts, str):
            raw_contexts = [raw_contexts]
        contexts = []
        for ctx in raw_contexts:
            try:
                contexts.append(SkillContext(ctx))
            except ValueError:
                pass
        if not contexts:
            contexts = [SkillContext.HEARTBEAT, SkillContext.CHAT]

        # Parse requires
        requires = metadata.get("requires", {})
        if not isinstance(requires, dict):
            requires = {}

        # Parse category
        raw_category = metadata.get("category", "other")
        try:
            category = SkillCategory(raw_category)
        except ValueError:
            category = SkillCategory.OTHER

        # Parse install methods
        raw_install = metadata.get("install", [])
        if not isinstance(raw_install, list):
            raw_install = []
        install_methods = [
            InstallMethod.from_dict(m) for m in raw_install if isinstance(m, dict)
        ]

        # Parse os_support
        raw_os = metadata.get("os_support", ["darwin", "linux"])
        if isinstance(raw_os, str):
            raw_os = [raw_os]

        # Parse bound_tools (J.4)
        raw_bound = metadata.get("bound_tools", [])
        if isinstance(raw_bound, str):
            raw_bound = [raw_bound]

        raw_provenance = metadata.get("provenance", {})
        if not isinstance(raw_provenance, dict):
            raw_provenance = {}

        raw_mcp = metadata.get("mcp")
        mcp_binding = (
            MCPBinding.from_dict(raw_mcp) if isinstance(raw_mcp, dict) else None
        )

        raw_blueprint = metadata.get("blueprint")
        if isinstance(raw_blueprint, str):
            # Inline JSON remains usable with the dependency-free fallback
            # frontmatter parser; normal YAML blocks arrive as dictionaries.
            try:
                import json

                parsed_blueprint = json.loads(raw_blueprint)
                raw_blueprint = (
                    parsed_blueprint if isinstance(parsed_blueprint, dict) else None
                )
            except (TypeError, ValueError):
                raw_blueprint = None
        blueprint = dict(raw_blueprint) if isinstance(raw_blueprint, dict) else None

        return cls(
            name=str(metadata.get("name", "")),
            description=str(metadata.get("description", "")),
            content=content.strip(),
            requires_tools=list(requires.get("tools", [])),
            requires_config=list(requires.get("config", [])),
            requires_bins=list(requires.get("bins", [])),
            requires_env=list(requires.get("env", [])),
            install_methods=install_methods,
            category=category,
            os_support=list(raw_os),
            bound_tools=list(raw_bound),
            aliases=[
                str(a).strip().lower()
                for a in (metadata.get("aliases") or [])
                if str(a).strip()
            ],
            activation_phrases=[
                str(phrase).strip().lower()
                for phrase in (metadata.get("activation_phrases") or [])
                if str(phrase).strip()
            ],
            contexts=contexts,
            source=source,
            provenance=dict(raw_provenance),
            mcp_binding=mcp_binding,
            blueprint=blueprint,
        )
