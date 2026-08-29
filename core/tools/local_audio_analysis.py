"""Optional approval-gated tool for device-local speaker diarization."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
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


async def _config(pool: Any, key: str, default: Any) -> Any:
    if pool is None:
        return default
    try:
        async with pool.acquire() as conn:
            value = await conn.fetchval("SELECT get_config($1)", key)
        return default if value is None else value
    except Exception:
        logger.debug("Could not read %s", key, exc_info=True)
        return default


async def _duration_seconds(path: Path) -> float | None:
    def probe() -> float | None:
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "json",
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if result.returncode != 0:
                return None
            return float(
                json.loads(result.stdout).get("format", {}).get("duration") or 0
            )
        except Exception:
            return None

    return await asyncio.to_thread(probe)


class AnalyzeLocalAudioHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="analyze_local_audio",
            description=(
                "Start or poll device-local speaker diarization for an audio file. "
                "Optionally label an existing Whisper JSON transcript and add explicitly "
                "marked acoustic heuristics. Audio is never uploaded. Results go to the "
                "Hexis cache; start the job, then poll status with the same audio_path."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["start", "status"],
                        "default": "start",
                    },
                    "audio_path": {
                        "type": "string",
                        "description": "Absolute path to an audio file on this device.",
                    },
                    "whisper_json_path": {
                        "type": "string",
                        "description": "Optional Whisper JSON containing timestamped segments.",
                    },
                    "emotion_heuristics": {
                        "type": "boolean",
                        "default": False,
                        "description": "Add coarse acoustic estimates clearly labeled heuristic_local.",
                    },
                },
                "required": ["audio_path"],
                "additionalProperties": False,
            },
            category=ToolCategory.INGEST,
            energy_cost=2,
            requires_approval=True,
            is_read_only=False,
            supports_parallel=False,
            optional=True,
            execution_timeout_seconds=60.0,
            allowed_contexts={ToolContext.CHAT, ToolContext.HEARTBEAT, ToolContext.MCP},
        )

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        from services.local_audio_analysis import (
            DEFAULT_MODEL,
            default_output_dir,
            resolve_hf_token,
            start_diarization_job,
            status_for,
        )

        pool = context.registry.pool if context.registry else None
        if not bool(await _config(pool, "audio_analysis.local.enabled", True)):
            return ToolResult.error_result(
                "Local audio analysis is off. Set audio_analysis.local.enabled=true, then retry.",
                ToolErrorType.DISABLED,
            )
        audio_text = str(arguments.get("audio_path") or "").strip()
        if not audio_text:
            return ToolResult.error_result(
                "audio_path is required", ToolErrorType.INVALID_PARAMS
            )
        audio = Path(audio_text).expanduser().resolve()
        action = str(arguments.get("action") or "start").strip().lower()
        output = default_output_dir(audio)
        if action == "status":
            status = status_for(output)
            return ToolResult.success_result(
                status,
                display_output=(
                    f"Audio analysis: {status.get('status')}; "
                    f"speakers={status.get('speaker_count')}; "
                    f"next={status.get('error') or status.get('output_dir')}"
                ),
            )
        if action != "start":
            return ToolResult.error_result(
                "action must be start or status", ToolErrorType.INVALID_PARAMS
            )
        if context.tool_context == ToolContext.HEARTBEAT and not bool(
            await _config(pool, "audio_analysis.local.allow_autonomous", False)
        ):
            return ToolResult.error_result(
                "Autonomous audio analysis is off. Use chat, or explicitly enable audio_analysis.local.allow_autonomous.",
                ToolErrorType.APPROVAL_REQUIRED,
            )
        if not audio.is_file():
            return ToolResult.error_result(
                f"Audio file not found: {audio}", ToolErrorType.FILE_NOT_FOUND
            )
        maximum = int(
            await _config(pool, "audio_analysis.local.max_duration_seconds", 7200)
        )
        duration = await _duration_seconds(audio)
        if duration is not None and duration > maximum:
            return ToolResult.error_result(
                f"Audio is {duration:.1f}s; the configured limit is {maximum}s.",
                ToolErrorType.PERMISSION_DENIED,
            )
        if not resolve_hf_token():
            return ToolResult.error_result(
                "HF_TOKEN is not set. Accept the configured pyannote model terms on Hugging Face, expose HF_TOKEN to Hexis, then retry.",
                ToolErrorType.AUTH_FAILED,
            )
        emotion_enabled = bool(
            await _config(pool, "audio_analysis.local.emotion.enabled", False)
        )
        emotion_requested = arguments.get("emotion_heuristics") is True
        model = str(
            await _config(
                pool,
                "audio_analysis.local.model",
                DEFAULT_MODEL,
            )
        )
        try:
            started = start_diarization_job(
                audio,
                whisper_json_path=arguments.get("whisper_json_path"),
                model_id=model,
                emotion_heuristics=emotion_enabled and emotion_requested,
            )
        except Exception as exc:
            logger.exception("Could not start local audio analysis")
            return ToolResult.error_result(
                f"Could not start local audio analysis: {exc}",
                ToolErrorType.EXECUTION_FAILED,
            )
        return ToolResult.success_result(
            {
                **started,
                "emotion_heuristics_requested": emotion_requested,
                "emotion_heuristics_effective": emotion_enabled and emotion_requested,
                "next_step": "Poll action=status with the same audio_path until status is completed or failed.",
            },
            display_output=(
                f"Audio analysis {started.get('status')} in {started.get('output_dir')}. "
                "Poll status with the same audio path."
            ),
        )


class TranscribeAudioHandler(ToolHandler):
    """Transcribe a local recording with the provider chosen in Settings."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="transcribe",
            description=(
                "Transcribe one local audio file with the voice-note provider selected "
                "in Settings. Local processing stays on-device; cloud processing is "
                "available only after its disclosure has been accepted."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "audio_path": {
                        "type": "string",
                        "description": "Absolute path to an audio file on this device.",
                    }
                },
                "required": ["audio_path"],
                "additionalProperties": False,
            },
            category=ToolCategory.INGEST,
            energy_cost=1,
            requires_approval=True,
            is_read_only=True,
            supports_parallel=False,
            optional=True,
            execution_timeout_seconds=None,
            allowed_contexts={ToolContext.CHAT, ToolContext.HEARTBEAT, ToolContext.MCP},
        )

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        from channels.media import Attachment
        from services.voice_notes import (
            SttConfig,
            load_stt_config,
            transcribe_attachment,
        )

        audio_text = str(arguments.get("audio_path") or "").strip()
        if not audio_text:
            return ToolResult.error_result(
                "audio_path is required", ToolErrorType.INVALID_PARAMS
            )
        audio = Path(audio_text).expanduser().resolve()
        if not audio.is_file():
            return ToolResult.error_result(
                f"Audio file not found: {audio}", ToolErrorType.FILE_NOT_FOUND
            )
        pool = context.registry.pool if context.registry else None
        cfg = SttConfig()
        if pool is not None:
            try:
                async with pool.acquire() as conn:
                    cfg = await load_stt_config(conn)
            except Exception as exc:
                return ToolResult.error_result(
                    f"Voice-note settings could not be loaded: {exc}",
                    ToolErrorType.MISSING_CONFIG,
                )
        if cfg.provider == "openai_whisper" and not cfg.cloud_disclosure_accepted:
            return ToolResult.error_result(
                "Cloud transcription has not been accepted. Open Settings → Voice notes, review the disclosure, and save the cloud choice.",
                ToolErrorType.MISSING_CONFIG,
            )
        attachment = Attachment(
            url="",
            filename=audio.name,
            mime_type=None,
            size=audio.stat().st_size,
            local_path=str(audio),
        )
        result = await transcribe_attachment(attachment, cfg=cfg)
        if not result.ok:
            detail = result.error_detail or result.outcome
            error_type = (
                ToolErrorType.MISSING_API_KEY
                if result.outcome == "failed_missing_credentials"
                else (
                    ToolErrorType.MISSING_DEPENDENCY
                    if "install" in detail.lower()
                    else ToolErrorType.EXECUTION_FAILED
                )
            )
            return ToolResult.error_result(
                f"Transcription failed: {detail}", error_type
            )
        return ToolResult.success_result(
            {
                "transcript": result.transcript,
                "provider": result.provider,
                "model": result.model,
                "duration_ms": result.duration_ms,
                "audio_path": str(audio),
            },
            display_output=result.transcript,
        )


def create_local_audio_analysis_tools() -> list[ToolHandler]:
    return [TranscribeAudioHandler(), AnalyzeLocalAudioHandler()]
