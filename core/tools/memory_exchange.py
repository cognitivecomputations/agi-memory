"""Skill-gated agent tools for Hexis Memory Exchange (HMX)."""

from __future__ import annotations

import copy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.hmx_files import (
    load_hmx_file,
    serialize_hmx_document,
    write_private_hmx_file,
)
from core.memory_exchange import (
    EXPORT_INTENTS,
    PROTECTED_SECTIONS,
    REDACTION_POLICIES,
    SUPPORTED_IMPORT_STRATEGIES,
    HmxPolicyError,
    HmxSchemaError,
    accept_staged_import,
    default_import_strategy,
    demote_staged_to_analysis,
    dry_run_hmx,
    export_hmx,
    import_hmx,
    normalize_replace_sections,
    modify_staged_import,
    pending_hmx_reviews,
    promote_analysis_to_staged,
    quote_staged_import,
    reject_staged_import,
)
from .base import (
    ToolCategory,
    ToolContext,
    ToolErrorType,
    ToolExecutionContext,
    ToolHandler,
    ToolResult,
    ToolSpec,
)

_AGENT_CONTEXTS = {ToolContext.CHAT, ToolContext.HEARTBEAT}


def _pool(context: ToolExecutionContext):
    return context.registry.pool if context.registry else None


def _missing_pool() -> ToolResult:
    return ToolResult.error_result(
        "HMX tools require an active Hexis database connection. Retry from a running Hexis chat or heartbeat.",
        ToolErrorType.MISSING_CONFIG,
    )


def _error_result(exc: Exception) -> ToolResult:
    if isinstance(exc, HmxSchemaError):
        return ToolResult.error_result(str(exc), ToolErrorType.INVALID_PARAMS)
    if isinstance(exc, HmxPolicyError):
        return ToolResult.error_result(str(exc), ToolErrorType.BOUNDARY_VIOLATION)
    return ToolResult.error_result(str(exc), ToolErrorType.EXECUTION_FAILED)


def _read_path(
    arguments: dict[str, Any], context: ToolExecutionContext
) -> Path | ToolResult:
    if not context.allow_file_read:
        return ToolResult.error_result(
            "HMX file reads are disabled in this tool context. Enable workspace file reads and retry.",
            ToolErrorType.PERMISSION_DENIED,
        )
    path = Path(context.resolve_path(str(arguments.get("path") or ""))).expanduser()
    if not context.is_path_allowed(str(path)):
        return ToolResult.error_result(
            f"HMX path is outside the allowed workspace: {path}",
            ToolErrorType.PATH_NOT_ALLOWED,
        )
    return path


def _write_path(path_value: str, context: ToolExecutionContext) -> Path | ToolResult:
    if not context.allow_file_write:
        return ToolResult.error_result(
            "HMX file writes are disabled in this tool context; omit output_path to return the exchange",
            ToolErrorType.PERMISSION_DENIED,
        )
    path = Path(context.resolve_path(path_value)).expanduser()
    if not context.is_path_allowed(str(path)):
        return ToolResult.error_result(
            f"HMX output path is outside the allowed workspace: {path}",
            ToolErrorType.PATH_NOT_ALLOWED,
        )
    return path


def _timestamp(value: Any, name: str) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise HmxPolicyError(f"{name} must be an ISO 8601 date or timestamp") from exc
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def _prepare_document(
    document: dict[str, Any], arguments: dict[str, Any]
) -> dict[str, Any]:
    prepared = copy.deepcopy(document)
    sections = prepared.get("sections")
    if isinstance(sections, dict):
        for section, flag in (
            ("identity", "skip_identity"),
            ("worldview", "skip_worldview"),
            ("narrative", "skip_narrative"),
        ):
            if arguments.get(flag):
                sections.pop(section, None)
    return prepared


def _path_parameter() -> dict[str, Any]:
    return {
        "type": "string",
        "description": "HMX JSON or JSONL path, relative to the active workspace when possible.",
    }


class ExportMemoriesHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="export_memories",
            description=(
                "Export an HMX memory exchange. Returns JSON/JSONL when output_path is omitted; "
                "file output is private and never overwrites unless overwrite=true."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "intent": {"type": "string", "enum": list(EXPORT_INTENTS)},
                    "output_path": {"type": "string"},
                    "format": {
                        "type": "string",
                        "enum": ["json", "jsonl"],
                        "default": "json",
                    },
                    "memory_types": {"type": "array", "items": {"type": "string"}},
                    "since": {"type": "string"},
                    "until": {"type": "string"},
                    "include_protected": {
                        "type": "array",
                        "items": {"type": "string", "enum": sorted(PROTECTED_SECTIONS)},
                    },
                    "include_raw": {"type": "boolean", "default": False},
                    "include_sensitive": {"type": "boolean", "default": False},
                    "include_config": {"type": "boolean", "default": False},
                    "include_in_flight_work": {"type": "boolean"},
                    "include_audit_records": {"type": "boolean"},
                    "redaction": {
                        "type": "string",
                        "enum": list(REDACTION_POLICIES),
                        "default": "none",
                    },
                    "overwrite": {"type": "boolean", "default": False},
                },
                "required": ["intent"],
            },
            category=ToolCategory.MEMORY,
            energy_cost=2,
            requires_approval=True,
            is_read_only=False,
            supports_parallel=False,
            allowed_contexts=_AGENT_CONTEXTS,
        )

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        pool = _pool(context)
        if not pool:
            return _missing_pool()
        output_path = arguments.get("output_path")
        resolved_output: Path | None = None
        if output_path:
            checked = _write_path(str(output_path), context)
            if isinstance(checked, ToolResult):
                return checked
            resolved_output = checked
        try:
            since = _timestamp(arguments.get("since"), "since")
            until = _timestamp(arguments.get("until"), "until")
            if since and until and since > until:
                raise HmxPolicyError("since must be earlier than or equal to until")
            async with pool.acquire() as conn:
                document = await export_hmx(
                    conn,
                    intent=str(arguments["intent"]),
                    include_protected=[
                        str(value).replace("-", "_")
                        for value in arguments.get("include_protected") or []
                    ],
                    include_raw_units=bool(arguments.get("include_raw", False)),
                    include_config=bool(arguments.get("include_config", False)),
                    include_in_flight_work=arguments.get("include_in_flight_work"),
                    include_audit_records=arguments.get("include_audit_records"),
                    types=[str(value) for value in arguments.get("memory_types") or []]
                    or None,
                    since=since,
                    until=until,
                    redaction_policy=str(arguments.get("redaction") or "none"),
                    include_sensitive=bool(arguments.get("include_sensitive", False)),
                )
            output_format = str(arguments.get("format") or "json")
            if resolved_output:
                content = serialize_hmx_document(document, output_format)
                written = write_private_hmx_file(
                    resolved_output,
                    content,
                    overwrite=bool(arguments.get("overwrite", False)),
                )
                return ToolResult.success_result(
                    {
                        "export_id": document["export_id"],
                        "path": str(written),
                        "format": output_format,
                        "statistics": document["statistics"],
                        "privacy": document["privacy"],
                        "warnings": document.get("export_warnings", []),
                    },
                    f"Exported HMX to {written}",
                )
            if output_format == "jsonl":
                return ToolResult.success_result(
                    {
                        "format": "jsonl",
                        "export_id": document["export_id"],
                        "content": serialize_hmx_document(document, "jsonl"),
                        "warnings": document.get("export_warnings", []),
                    },
                    f"Exported HMX {document['export_id']} as JSONL",
                )
            return ToolResult.success_result(
                document, f"Exported HMX {document['export_id']}"
            )
        except Exception as exc:
            return _error_result(exc)


class _ImportFileHandler(ToolHandler):
    def _load(self, arguments: dict[str, Any], context: ToolExecutionContext):
        checked = _read_path(arguments, context)
        if isinstance(checked, ToolResult):
            return checked
        try:
            return _prepare_document(load_hmx_file(checked), arguments)
        except Exception as exc:
            return _error_result(exc)


class ImportDryRunHandler(_ImportFileHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="import_dry_run",
            description="Validate an HMX file and forecast policy, conflicts, counts, and embedding work without mutation.",
            parameters={
                "type": "object",
                "properties": {
                    "path": _path_parameter(),
                    "strategy": {
                        "type": "string",
                        "enum": list(SUPPORTED_IMPORT_STRATEGIES),
                    },
                    "skip_identity": {"type": "boolean", "default": False},
                    "skip_worldview": {"type": "boolean", "default": False},
                    "skip_narrative": {"type": "boolean", "default": False},
                    "retry_failed_work": {"type": "boolean", "default": False},
                    "replace_sections": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": sorted(PROTECTED_SECTIONS),
                        },
                        "uniqueItems": True,
                    },
                    "trust_matching_lineage_label": {
                        "type": "boolean",
                        "default": False,
                    },
                },
                "required": ["path"],
            },
            category=ToolCategory.MEMORY,
            energy_cost=1,
            is_read_only=True,
            allowed_contexts=_AGENT_CONTEXTS,
        )

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        document = self._load(arguments, context)
        if isinstance(document, ToolResult):
            return document
        pool = _pool(context)
        if not pool:
            return _missing_pool()
        try:
            intent = str(document.get("export_intent") or "")
            strategy = str(arguments.get("strategy") or default_import_strategy(intent))
            async with pool.acquire() as conn:
                result = await dry_run_hmx(
                    conn,
                    document,
                    strategy=strategy,
                    retry_failed_work=bool(arguments.get("retry_failed_work", False)),
                    replace_sections=normalize_replace_sections(
                        arguments.get("replace_sections")
                    ),
                    allow_locally_trusted_lineage=bool(
                        arguments.get("trust_matching_lineage_label", False)
                    ),
                )
            return ToolResult.success_result(
                asdict(result),
                f"HMX dry run: {'permitted' if result.can_import else 'blocked'}",
            )
        except Exception as exc:
            return _error_result(exc)


class ImportMemoriesHandler(_ImportFileHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="import_memories",
            description=(
                "Import an HMX file using additive, authoritative, deliberative, or "
                "analysis-only storage. confirm_intent must exactly match the file. "
                "Authoritative import requires explicit replace_sections and a rationale. "
                "Run import_dry_run first."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": _path_parameter(),
                    "strategy": {
                        "type": "string",
                        "enum": list(SUPPORTED_IMPORT_STRATEGIES),
                    },
                    "confirm_intent": {
                        "type": "string",
                        "enum": list(EXPORT_INTENTS),
                    },
                    "skip_identity": {"type": "boolean", "default": False},
                    "skip_worldview": {"type": "boolean", "default": False},
                    "skip_narrative": {"type": "boolean", "default": False},
                    "retry_failed_work": {"type": "boolean", "default": False},
                    "replace_sections": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": sorted(PROTECTED_SECTIONS),
                        },
                        "uniqueItems": True,
                    },
                    "replacement_rationale": {"type": "string"},
                    "trust_matching_lineage_label": {
                        "type": "boolean",
                        "default": False,
                    },
                },
                "required": ["path", "confirm_intent"],
            },
            category=ToolCategory.MEMORY,
            energy_cost=3,
            requires_approval=True,
            is_read_only=False,
            supports_parallel=False,
            allowed_contexts=_AGENT_CONTEXTS,
        )

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        document = self._load(arguments, context)
        if isinstance(document, ToolResult):
            return document
        intent = str(document.get("export_intent") or "")
        if str(arguments.get("confirm_intent") or "") != intent:
            return ToolResult.error_result(
                f"Intent confirmation mismatch: file declares {intent!r}",
                ToolErrorType.BOUNDARY_VIOLATION,
            )
        pool = _pool(context)
        if not pool:
            return _missing_pool()
        try:
            strategy = str(arguments.get("strategy") or default_import_strategy(intent))
            async with pool.acquire() as conn:
                retry_failed_work = bool(arguments.get("retry_failed_work", False))
                forecast = await dry_run_hmx(
                    conn,
                    document,
                    strategy=strategy,
                    retry_failed_work=retry_failed_work,
                    replace_sections=normalize_replace_sections(
                        arguments.get("replace_sections")
                    ),
                    allow_locally_trusted_lineage=bool(
                        arguments.get("trust_matching_lineage_label", False)
                    ),
                )
                if not forecast.can_import:
                    return ToolResult(
                        success=False,
                        output=asdict(forecast),
                        error=(
                            "HMX import preflight blocked this mutation. Inspect the returned "
                            "conflicts and warnings, choose a supported strategy or narrower "
                            "section scope, then retry."
                        ),
                        error_type=ToolErrorType.BOUNDARY_VIOLATION,
                    )
                result = await import_hmx(
                    conn,
                    document,
                    strategy=strategy,
                    retry_failed_work=retry_failed_work,
                    replace_sections=normalize_replace_sections(
                        arguments.get("replace_sections")
                    ),
                    replacement_rationale=arguments.get("replacement_rationale"),
                    allow_locally_trusted_lineage=bool(
                        arguments.get("trust_matching_lineage_label", False)
                    ),
                )
            return ToolResult.success_result(
                asdict(result), f"HMX import completed with {strategy}"
            )
        except Exception as exc:
            return _error_result(exc)


class ImportReviewHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="import_review",
            description="List pending deliberative HMX records with their conflicts and source context.",
            parameters={"type": "object", "properties": {}},
            category=ToolCategory.MEMORY,
            energy_cost=0,
            is_read_only=True,
            allowed_contexts=_AGENT_CONTEXTS,
        )

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        pool = _pool(context)
        if not pool:
            return _missing_pool()
        try:
            async with pool.acquire() as conn:
                result = await pending_hmx_reviews(conn)
            return ToolResult.success_result(
                result, f"{result['total']} HMX record(s) pending review"
            )
        except Exception as exc:
            return _error_result(exc)


class _StagingDecisionHandler(ToolHandler):
    tool_name = ""
    decision = ""
    description = ""
    parameters: dict[str, Any] = {}
    required_parameters: tuple[str, ...] = ()

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {"staging_id": {"type": "string"}, **self.parameters},
                "required": ["staging_id", *self.required_parameters],
            },
            category=ToolCategory.MEMORY,
            energy_cost=2,
            requires_approval=True,
            is_read_only=False,
            supports_parallel=False,
            allowed_contexts=_AGENT_CONTEXTS,
        )


class ImportAcceptHandler(_StagingDecisionHandler):
    tool_name = "import_accept"
    description = "Accept one pending HMX record after deliberative review; protected active-state policy still applies."
    parameters = {"rationale": {"type": "string"}}

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        pool = _pool(context)
        if not pool:
            return _missing_pool()
        try:
            async with pool.acquire() as conn:
                result = await accept_staged_import(
                    conn,
                    str(arguments["staging_id"]),
                    rationale=arguments.get("rationale"),
                )
            return ToolResult.success_result(
                asdict(result), "Accepted staged HMX record"
            )
        except Exception as exc:
            return _error_result(exc)


class ImportRejectHandler(_StagingDecisionHandler):
    tool_name = "import_reject"
    description = "Reject one pending HMX record while retaining its review history and rationale."
    parameters = {"rationale": {"type": "string"}}
    required_parameters = ("rationale",)

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        pool = _pool(context)
        if not pool:
            return _missing_pool()
        try:
            async with pool.acquire() as conn:
                result = await reject_staged_import(
                    conn,
                    str(arguments["staging_id"]),
                    rationale=str(arguments["rationale"]),
                )
            return ToolResult.success_result(
                asdict(result), "Rejected staged HMX record"
            )
        except Exception as exc:
            return _error_result(exc)


class ImportModifyHandler(_StagingDecisionHandler):
    tool_name = "import_modify"
    description = "Modify a pending HMX record before acceptance and append material-change provenance."
    parameters = {
        "changes": {"type": "object"},
        "modification_kind": {"type": "string"},
        "rationale": {"type": "string"},
    }
    required_parameters = ("changes", "modification_kind", "rationale")

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        pool = _pool(context)
        if not pool:
            return _missing_pool()
        try:
            async with pool.acquire() as conn:
                result = await modify_staged_import(
                    conn,
                    str(arguments["staging_id"]),
                    dict(arguments["changes"]),
                    modification_kind=str(arguments["modification_kind"]),
                    rationale=str(arguments["rationale"]),
                )
            return ToolResult.success_result(
                asdict(result), "Modified staged HMX record"
            )
        except Exception as exc:
            return _error_result(exc)


class ImportQuoteHandler(_StagingDecisionHandler):
    tool_name = "import_quote"
    description = "Preserve a pending HMX record as archived foreign quoted context, not active memory."
    parameters = {"rationale": {"type": "string"}}
    required_parameters = ("rationale",)

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        pool = _pool(context)
        if not pool:
            return _missing_pool()
        try:
            async with pool.acquire() as conn:
                result = await quote_staged_import(
                    conn,
                    str(arguments["staging_id"]),
                    rationale=str(arguments["rationale"]),
                )
            return ToolResult.success_result(
                asdict(result), "Archived HMX record as quoted context"
            )
        except Exception as exc:
            return _error_result(exc)


class PromoteToStagedHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="promote_to_staged",
            description="Copy one isolated analysis record into deliberative staging without copying embeddings.",
            parameters={
                "type": "object",
                "properties": {
                    "analysis_id": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["analysis_id", "rationale"],
            },
            category=ToolCategory.MEMORY,
            energy_cost=1,
            requires_approval=True,
            is_read_only=False,
            supports_parallel=False,
            allowed_contexts=_AGENT_CONTEXTS,
        )

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        pool = _pool(context)
        if not pool:
            return _missing_pool()
        try:
            async with pool.acquire() as conn:
                staging_id = await promote_analysis_to_staged(
                    conn,
                    str(arguments["analysis_id"]),
                    rationale=str(arguments["rationale"]),
                )
            return ToolResult.success_result(
                {"decision": "promoted", "staging_id": staging_id},
                "Promoted analysis record to staging",
            )
        except Exception as exc:
            return _error_result(exc)


class DemoteToAnalysisHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="demote_to_analysis",
            description="Move one pending staged record into isolated analysis-only storage with rationale.",
            parameters={
                "type": "object",
                "properties": {
                    "staging_id": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["staging_id", "rationale"],
            },
            category=ToolCategory.MEMORY,
            energy_cost=1,
            requires_approval=True,
            is_read_only=False,
            supports_parallel=False,
            allowed_contexts=_AGENT_CONTEXTS,
        )

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        pool = _pool(context)
        if not pool:
            return _missing_pool()
        try:
            async with pool.acquire() as conn:
                analysis_id = await demote_staged_to_analysis(
                    conn,
                    str(arguments["staging_id"]),
                    rationale=str(arguments["rationale"]),
                )
            return ToolResult.success_result(
                {"decision": "demoted", "analysis_id": analysis_id},
                "Demoted staged record to analysis",
            )
        except Exception as exc:
            return _error_result(exc)


def create_memory_exchange_tools() -> list[ToolHandler]:
    from .protected_replacement import create_protected_replacement_tools

    return [
        ExportMemoriesHandler(),
        ImportMemoriesHandler(),
        ImportDryRunHandler(),
        ImportReviewHandler(),
        ImportAcceptHandler(),
        ImportRejectHandler(),
        ImportModifyHandler(),
        ImportQuoteHandler(),
        PromoteToStagedHandler(),
        DemoteToAnalysisHandler(),
    ] + create_protected_replacement_tools()
