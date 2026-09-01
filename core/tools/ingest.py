"""
Hexis Tools System - Ingestion Tools

Tools for content ingestion: fast (shallow), slow (conscious RLM), hybrid.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from .base import (
    ToolCategory,
    ToolContext,
    ToolErrorType,
    ToolExecutionContext,
    ToolHandler,
    ToolResult,
    ToolSpec,
)

logger = logging.getLogger(__name__)


async def _build_ingest_config(pool, **overrides: Any) -> "Config":
    """Build an ingestion Config with fully-resolved LLM credentials.

    Uses :func:`core.llm_config.resolve_llm_config` so that OAuth/token-refresh
    providers (Codex, Copilot, Gemini CLI, etc.) work correctly.
    """
    from core.agent_api import db_dsn_from_env
    from core.llm_config import resolve_llm_config
    from services.ingest import Config

    from services.ingest.config import load_ingest_settings

    dsn = db_dsn_from_env()
    llm_config = await resolve_llm_config(pool, "llm.chat", fallback_key="llm")

    defaults: dict[str, Any] = {
        "dsn": dsn,
        "llm_config": llm_config,
    }
    # DB-owned policy (#91); explicit overrides still win.
    async with pool.acquire() as conn:
        defaults.update(await load_ingest_settings(conn))
    defaults.update(overrides)
    return Config(**defaults)


def _acquisition_override(arguments: dict[str, Any], context: "ToolExecutionContext") -> dict[str, Any]:
    """Acquisition provenance: heartbeat-initiated ingestion is the agent's
    own choice ('agent'); chat/MCP ingestion acts on the user's behalf
    ('user'). Drives retention policy — user sources never auto-fade."""
    overrides: dict[str, Any] = {}
    tool_context = getattr(context, "tool_context", None)
    overrides["acquisition"] = (
        "agent" if tool_context is ToolContext.HEARTBEAT else "user"
    )
    keep_reason = str(arguments.get("keep_reason") or "").strip()
    if keep_reason:
        overrides["acquired_reason"] = keep_reason
    return overrides


def _pipeline_result(pipeline: Any) -> dict[str, Any]:
    result = getattr(pipeline, "last_result", None)
    return dict(result) if isinstance(result, dict) else {}


class FastIngestHandler(ToolHandler):
    """Fast (shallow) content ingestion.

    Chunks content, extracts facts via LLM, creates semantic memories
    with basic graph linking. No deep reasoning -- quick and cheap.
    """

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="fast_ingest",
            description=(
                "Quickly ingest a file into memory. Chunks the content, extracts key "
                "facts, and stores them as semantic memories with basic graph links. "
                "Use for content that doesn't require deep analysis or when energy is limited."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to ingest.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional title for the content.",
                    },
                },
                "required": ["path"],
            },
            category=ToolCategory.INGEST,
            energy_cost=2,
            is_read_only=False,
        )

    def validate(self, arguments: dict[str, Any]) -> list[str]:
        errors = []
        path = arguments.get("path", "")
        if not path or not str(path).strip():
            errors.append("path cannot be empty")
        return errors

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        from services.ingest import IngestionMode, IngestionPipeline

        path_str = str(arguments["path"]).strip()
        file_path = Path(path_str)

        if not file_path.exists():
            return ToolResult.error_result(
                f"File not found: {path_str}",
                ToolErrorType.FILE_NOT_FOUND,
            )

        pool = context.registry.pool
        config = await _build_ingest_config(pool, mode=IngestionMode.FAST, **_acquisition_override(arguments, context))
        pipeline = IngestionPipeline(config)

        try:
            count = await pipeline.ingest_file(file_path)
            details = _pipeline_result(pipeline)

            output = {
                "memories_created": count,
                "path": path_str,
                "mode": "fast",
                **details,
            }
            return ToolResult.success_result(
                output,
                display_output=f"Fast ingested {path_str}: {count} memories created.",
            )
        except Exception as e:
            logger.error("fast_ingest failed: %s", e)
            return ToolResult.error_result(
                f"Fast ingestion failed: {e}",
                ToolErrorType.EXECUTION_FAILED,
            )
        finally:
            await pipeline.close()


class SlowIngestHandler(ToolHandler):
    """Slow (conscious) content ingestion via RLM loop.

    Runs a mini-RLM loop per chunk: searches related memories, checks
    worldview, forms emotional reaction, writes analysis, decides
    acceptance level. Creates rich memories with emotional context,
    deep graph connections, and contested/questioned flags.
    """

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="slow_ingest",
            description=(
                "Deeply and consciously ingest a file into memory. Each chunk is "
                "processed through a reasoning loop: you'll search related memories, "
                "compare against your worldview, form emotional reactions, and decide "
                "whether to accept, contest, or question each piece of knowledge. "
                "Creates richly connected memories. Use for important content that "
                "deserves careful consideration."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to ingest.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional title for the content.",
                    },
                },
                "required": ["path"],
            },
            category=ToolCategory.INGEST,
            energy_cost=5,
            is_read_only=False,
        )

    def validate(self, arguments: dict[str, Any]) -> list[str]:
        errors = []
        path = arguments.get("path", "")
        if not path or not str(path).strip():
            errors.append("path cannot be empty")
        return errors

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        from services.ingest import IngestionMode, IngestionPipeline

        path_str = str(arguments["path"]).strip()
        file_path = Path(path_str)

        if not file_path.exists():
            return ToolResult.error_result(
                f"File not found: {path_str}",
                ToolErrorType.FILE_NOT_FOUND,
            )

        pool = context.registry.pool
        config = await _build_ingest_config(pool, mode=IngestionMode.SLOW, **_acquisition_override(arguments, context))
        pipeline = IngestionPipeline(config)

        try:
            count = await pipeline.ingest_file(file_path)
            details = _pipeline_result(pipeline)

            output = {
                "memories_created": count,
                "path": path_str,
                "mode": "slow",
                **details,
            }
            return ToolResult.success_result(
                output,
                display_output=f"Slow ingested {path_str}: {count} memories created.",
            )
        except Exception as e:
            logger.error("slow_ingest failed: %s", e)
            return ToolResult.error_result(
                f"Slow ingestion failed: {e}",
                ToolErrorType.EXECUTION_FAILED,
            )
        finally:
            await pipeline.close()


class HybridIngestHandler(ToolHandler):
    """Hybrid content ingestion: fast first pass, slow on high-signal chunks.

    Does a quick extraction pass to score all chunks, then runs the full
    RLM conscious reading loop only on chunks that are high-importance,
    contradict existing worldview, or relate to active goals.
    """

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="hybrid_ingest",
            description=(
                "Ingest a file using a hybrid approach: quickly scan all chunks to "
                "identify which ones are most important or potentially contradictory, "
                "then deeply process only those high-signal chunks through conscious "
                "reading. A good balance between thoroughness and energy efficiency."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to ingest.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional title for the content.",
                    },
                },
                "required": ["path"],
            },
            category=ToolCategory.INGEST,
            energy_cost=3,
            is_read_only=False,
        )

    def validate(self, arguments: dict[str, Any]) -> list[str]:
        errors = []
        path = arguments.get("path", "")
        if not path or not str(path).strip():
            errors.append("path cannot be empty")
        return errors

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        from services.ingest import IngestionMode, IngestionPipeline

        path_str = str(arguments["path"]).strip()
        file_path = Path(path_str)

        if not file_path.exists():
            return ToolResult.error_result(
                f"File not found: {path_str}",
                ToolErrorType.FILE_NOT_FOUND,
            )

        pool = context.registry.pool
        config = await _build_ingest_config(pool, mode=IngestionMode.HYBRID, **_acquisition_override(arguments, context))
        pipeline = IngestionPipeline(config)

        try:
            count = await pipeline.ingest_file(file_path)
            details = _pipeline_result(pipeline)

            output = {
                "memories_created": count,
                "path": path_str,
                "mode": "hybrid",
                **details,
            }
            return ToolResult.success_result(
                output,
                display_output=f"Hybrid ingested {path_str}: {count} memories created.",
            )
        except Exception as e:
            logger.error("hybrid_ingest failed: %s", e)
            return ToolResult.error_result(
                f"Hybrid ingestion failed: {e}",
                ToolErrorType.EXECUTION_FAILED,
            )
        finally:
            await pipeline.close()


class GitIngestHandler(ToolHandler):
    """Ingest a GitHub repository by cloning and processing its files."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="git_ingest",
            description=(
                "Clone a GitHub repository and ingest its contents into memory. "
                "Accepts a full URL (https://github.com/owner/repo) or shorthand "
                "(owner/repo). Filters out common junk directories (.git, node_modules, etc.)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "GitHub repo URL or owner/repo shorthand.",
                    },
                    "branch": {
                        "type": "string",
                        "description": "Branch to clone (default: repo default branch).",
                    },
                    "keep_reason": {
                        "type": "string",
                        "description": (
                            "Why you decided to keep this source (recorded as "
                            "acquisition provenance; helps future retention decisions)."
                        ),
                    },
                },
                "required": ["url"],
            },
            category=ToolCategory.INGEST,
            energy_cost=4,
            is_read_only=False,
        )

    def validate(self, arguments: dict[str, Any]) -> list[str]:
        errors = []
        url = arguments.get("url", "")
        if not url or not str(url).strip():
            errors.append("url cannot be empty")
        return errors

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        import shutil
        import subprocess
        import tempfile

        from services.ingest import IngestionMode, IngestionPipeline

        repo = str(arguments["url"]).strip()
        if "/" in repo and not repo.startswith("http"):
            repo = f"https://github.com/{repo}"

        branch = arguments.get("branch")

        pool = context.registry.pool
        config = await _build_ingest_config(pool, mode=IngestionMode.FAST, **_acquisition_override(arguments, context))
        pipeline = IngestionPipeline(config)
        tmpdir = tempfile.mkdtemp(prefix="hexis_git_")

        try:
            cmd = ["git", "clone", "--depth", "1"]
            if branch:
                cmd.extend(["--branch", str(branch)])
            cmd.extend([repo, tmpdir])

            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: subprocess.check_call(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
                ),
            )

            count = await pipeline.ingest_directory(
                Path(tmpdir),
                recursive=True,
                exclude_dirs=IngestionPipeline.GIT_IGNORE_DIRS,
            )
            details = _pipeline_result(pipeline)

            output = {
                "memories_created": count,
                "repo": repo,
                "branch": branch,
                **details,
            }
            return ToolResult.success_result(
                output,
                display_output=f"Ingested {repo}: {count} memories created.",
            )
        except subprocess.CalledProcessError as e:
            logger.error("git_ingest clone failed: %s", e)
            return ToolResult.error_result(
                f"Git clone failed: {e}",
                ToolErrorType.EXECUTION_FAILED,
            )
        except Exception as e:
            logger.error("git_ingest failed: %s", e)
            return ToolResult.error_result(
                f"Git ingestion failed: {e}",
                ToolErrorType.EXECUTION_FAILED,
            )
        finally:
            await pipeline.close()
            shutil.rmtree(tmpdir, ignore_errors=True)


def _detect_url_source_type(url: str) -> str:
    """Detect source type from a URL for routing to the right extractor.

    Returns one of: 'youtube', 'pdf', 'rss', 'web'.
    """
    from services.ingest import YouTubeTranscriptReader

    if YouTubeTranscriptReader.can_handle(url):
        return "youtube"

    # Check URL path extension
    from urllib.parse import urlparse
    parsed = urlparse(url)
    path_lower = parsed.path.lower()
    if path_lower.endswith(".pdf") or path_lower.startswith("/pdf/"):
        return "pdf"
    if path_lower.endswith((".rss", ".atom", ".xml")):
        # Heuristic: RSS/Atom feeds often have these extensions
        return "rss"

    return "web"


class URLIngestHandler(ToolHandler):
    """Queue web content from a URL for durable background ingestion.

    The worker performs extraction and memory creation. Keeping this tool fast
    prevents chat/heartbeat turns from blocking on large PDFs or slow sites.
    """

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="url_ingest",
            description=(
                "Queue a URL for background ingestion into memory. Use after "
                "you have fetched/read a web source worth keeping; do not use "
                "this for immediate analysis. The worker later fetches web pages "
                "(HTML), PDFs, YouTube transcripts, and RSS/Atom feeds, preserves "
                "the source, creates memories, and loads single-source intake to "
                "the RecMem desk."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to fetch and ingest (http:// or https://).",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["fast", "slow", "hybrid"],
                        "default": "fast",
                        "description": (
                            "Ingestion depth: 'fast' extracts key facts quickly, "
                            "'slow' runs deep conscious reasoning on each chunk, "
                            "'hybrid' triages then deep-reads important chunks."
                        ),
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional title for the content (auto-detected if omitted).",
                    },
                    "keep_reason": {
                        "type": "string",
                        "description": (
                            "Why you decided to keep this web source (recorded as "
                            "acquisition provenance; helps future retention decisions)."
                        ),
                    },
                },
                "required": ["url"],
            },
            category=ToolCategory.INGEST,
            energy_cost=3,
            is_read_only=False,
        )

    def validate(self, arguments: dict[str, Any]) -> list[str]:
        errors = []
        url = arguments.get("url", "")
        if not url or not str(url).strip():
            errors.append("url cannot be empty")
        elif not (str(url).startswith("http://") or str(url).startswith("https://")):
            errors.append("url must start with http:// or https://")
        else:
            from core.tools.web import _validate_url_host
            errors.extend(_validate_url_host(str(url)))
        return errors

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        url = str(arguments["url"]).strip()
        mode_str = str(arguments.get("mode") or "fast").lower()
        if mode_str not in ("fast", "slow", "hybrid"):
            return ToolResult.error_result(
                "mode must be fast, slow, or hybrid",
                ToolErrorType.INVALID_PARAMS,
            )

        registry = getattr(context, "registry", None)
        pool = getattr(registry, "pool", None)
        if pool is None:
            return ToolResult.error_result(
                "URL ingestion queue is unavailable: database pool is not configured",
                ToolErrorType.MISSING_CONFIG,
            )

        source_type = _detect_url_source_type(url)
        payload: dict[str, Any] = {
            "url": url,
            "title": str(arguments.get("title") or "").strip() or None,
            "mode": mode_str,
            "source_type": source_type,
        }
        payload.update(_acquisition_override(arguments, context))

        content_hash = f"url:{hashlib.sha256(url.encode('utf-8')).hexdigest()}"
        try:
            async with pool.acquire() as conn:
                job_id = await conn.fetchval(
                    "SELECT enqueue_ingestion_job('url', $1::jsonb, NULL, $2)",
                    json.dumps(payload),
                    content_hash,
                )
        except Exception as e:
            logger.error("url_ingest enqueue failed: %s", e)
            return ToolResult.error_result(
                f"URL ingestion could not be queued: {e}",
                ToolErrorType.EXECUTION_FAILED,
            )

        output = {
            "accepted": True,
            "job_id": str(job_id),
            "url": url,
            "mode": mode_str,
            "source_type": source_type,
            "content_hash": content_hash,
        }
        return ToolResult.success_result(
            output,
            display_output=(
                f"Queued URL ingestion for {url} ({source_type}, {mode_str}); "
                f"job {job_id}. The maintenance worker will fetch, preserve, "
                "ingest, and desk-load the source."
            ),
        )


def create_ingest_tools() -> list[ToolHandler]:
    """Create all ingestion tool handlers."""
    return [
        FastIngestHandler(),
        SlowIngestHandler(),
        HybridIngestHandler(),
        GitIngestHandler(),
        URLIngestHandler(),
    ]
