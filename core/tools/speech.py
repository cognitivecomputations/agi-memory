"""Speech output tool backed by the configured local synthesizer."""

from __future__ import annotations

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


class SpeakHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="speak",
            description=(
                "Render text as local speech when the user asks to hear something aloud. "
                "The configured local sidecar receives the text; the metadata audit does not."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "minLength": 1,
                        "description": "The exact text to speak.",
                    }
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            category=ToolCategory.EXTERNAL,
            energy_cost=1,
            requires_approval=False,
            is_read_only=False,
            supports_parallel=False,
            optional=True,
            allowed_contexts={ToolContext.CHAT},
        )

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        if context.registry is None:
            return ToolResult.error_result(
                "speak requires a database-backed tool registry",
                ToolErrorType.EXECUTION_FAILED,
            )
        text = str(arguments.get("text") or "").strip()
        if not text:
            return ToolResult.error_result(
                "text is required", ToolErrorType.INVALID_PARAMS
            )

        from services.speech import (
            load_tts_config,
            store_synthesis_output,
            synthesize_text,
        )

        async with context.registry.pool.acquire() as conn:
            cfg = await load_tts_config(conn)
        result = await synthesize_text(
            context.registry.pool,
            text,
            source=f"tool:{context.surface or 'chat'}",
            cfg=cfg,
        )
        if not result.ok:
            error_type = (
                ToolErrorType.DISABLED
                if result.outcome == "skipped_disabled"
                else ToolErrorType.MISSING_CONFIG
                if result.outcome in {"failed_endpoint", "failed_unsupported_provider"}
                else ToolErrorType.EXECUTION_FAILED
            )
            return ToolResult.error_result(
                result.error_detail or "speech synthesis failed", error_type
            )
        output_id = await store_synthesis_output(
            context.registry.pool,
            result,
            ttl_minutes=cfg.output_ttl_minutes,
            metadata={
                "call_id": context.call_id,
                "session_id": context.session_id,
                "surface": context.surface,
            },
        )
        audio_url = f"/api/voice/audio/{output_id}"
        await context.emit_event(
            "ui",
            {
                "kind": "speech",
                "id": output_id,
                "audio_url": audio_url,
                "mime_type": result.mime_type,
                "provider": result.provider,
                "model": result.model,
            },
        )
        return ToolResult.success_result(
            {
                "status": "ready",
                "audio_id": output_id,
                "audio_url": audio_url,
                "mime_type": result.mime_type,
                "provider": result.provider,
                "model": result.model,
                "expires_in_minutes": cfg.output_ttl_minutes,
            },
            display_output="Speech is ready in the conversation audio player.",
        )


def create_speech_tools() -> list[ToolHandler]:
    return [SpeakHandler()]
