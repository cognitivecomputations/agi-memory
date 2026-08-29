from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import socket
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any
import asyncpg

# Import order is intentional: this loads the explicitly selected environment
# before modules below derive process-level connection constants.
from services import worker_environment as _worker_environment  # noqa: F401

from core.agent_api import db_dsn_from_env
from core.gateway import EventSource, Gateway, GatewayConsumer, GatewayEvent
from core.rabbitmq_bridge import RabbitMQBridge
from core.state import (
    has_pending_inbound_disposition_wake,
    is_agent_terminated,
    mark_subconscious_decider_run,
    recompute_cron_next_runs,
    run_heartbeat,
    run_maintenance_if_due,
    run_scheduled_tasks,
    should_run_subconscious_decider,
)
from services.external_calls import ExternalCallProcessor
from services.heartbeat_agentic import finalize_heartbeat, run_agentic_heartbeat
from services.heartbeat_runner import execute_heartbeat_decision
from services.hmx_reembedding import run_hmx_reembed_step
from services.memory_embeddings import run_memory_embed_step
from services.source_chunks import run_source_chunk_embed_step
from services.recmem import (
    run_recmem_consolidation_step,
    run_recmem_embed_step,
    run_recmem_route_step,
    run_recmem_sweep_step,
)
from services.extraction import run_conscious_extraction_step
from services.channel_backfill import run_channel_backfill_step
from services.connector_cognition import (
    run_connector_importance_step,
    run_user_model_synthesis_step,
)
from services.gmail_backfill import run_gmail_backfill_step
from services.ambient_responsibilities import run_ambient_responsibility_step
from services.summarization import run_memory_summarization_step
from services.skill_improvement import (
    apply_approved_learning_skill_step,
    run_skill_improvement_review_step,
)
from services.reconsolidation import run_reconsolidation_step
from services.retention_surface import publish_retention_surface
from services.subconscious import run_subconscious_decider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("heartbeat_worker")

POLL_INTERVAL = float(os.getenv("WORKER_POLL_INTERVAL", 1.0))
MAX_RETRIES = int(os.getenv("WORKER_MAX_RETRIES", 3))
WORKER_STORM_MAX_STARTS = int(os.getenv("WORKER_STORM_MAX_STARTS", 5))
WORKER_STORM_WINDOW_SECONDS = int(os.getenv("WORKER_STORM_WINDOW_SECONDS", 120))
WORKER_STORM_BACKOFF_CAP_SECONDS = int(
    os.getenv("WORKER_STORM_BACKOFF_CAP_SECONDS", 300)
)


def _json_payload(value: Any) -> str:
    return json.dumps(value if value is not None else {}, default=str)


def _result_has_work(result: Any) -> bool:
    if result is None:
        return False
    if isinstance(result, bool):
        return result
    if isinstance(result, int):
        return result > 0
    if isinstance(result, (list, tuple, set)):
        return bool(result)
    if isinstance(result, dict):
        return not bool(result.get("skipped"))
    return True


async def channel_adapters_active(conn: asyncpg.Connection) -> bool:
    """Should outbox messages be published to RabbitMQ at all?

    Only when a channel adapter exists to consume them — with no adapter,
    user-bound messages pile up in the broker forever with every status
    surface green (#107). In a channels-less deployment the web_inbox tee IS
    the delivery. Fail open: better a queued broker copy than a lost channel
    delivery.
    """
    try:
        return bool(
            await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM channel_adapter_runtime WHERE configured OR running)"
            )
        )
    except Exception:
        return True


async def tee_outbox_to_web_inbox(
    conn: asyncpg.Connection, messages: list[dict]
) -> int:
    """Make dashboard delivery independent of the channel worker.

    ChannelOutboxConsumer also tees RabbitMQ messages into web_inbox. This
    local copy uses the same envelope id, so redelivery through RabbitMQ is
    idempotent instead of duplicative.
    """
    if not messages:
        return 0
    enabled = await conn.fetchval(
        "SELECT COALESCE(get_config_bool('channel.web_inbox.enabled'), TRUE)"
    )
    if not enabled:
        return 0

    delivered = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        payload = msg.get("payload") if isinstance(msg.get("payload"), dict) else {}
        delivery_info = payload.get("delivery") or msg.get("delivery")
        if isinstance(delivery_info, dict) and delivery_info.get("mode") == "silent":
            continue
        body = {
            "id": msg.get("message_id") or msg.get("id"),
            "kind": msg.get("kind"),
            "payload": payload,
        }
        if msg.get("delivery") is not None:
            body["delivery"] = msg.get("delivery")
        if msg.get("task_name") is not None:
            body["task_name"] = msg.get("task_name")
        web_id = await conn.fetchval(
            "SELECT web_inbox_deliver($1::jsonb)",
            json.dumps(body, default=str),
        )
        if web_id:
            delivered += 1
    return delivered


def _worker_metadata() -> dict[str, Any]:
    return {
        "process_id": os.getpid(),
        "host_name": socket.gethostname(),
        "command": "hexis-worker",
    }


async def _recover_worker_runtime(pool: asyncpg.Pool) -> None:
    try:
        async with pool.acquire() as conn:
            await conn.fetchval(
                "SELECT recover_interrupted_worker_runs($1::interval)", "10 minutes"
            )
    except Exception:
        logger.debug("worker runtime recovery unavailable", exc_info=True)


async def _record_worker_start_and_maybe_backoff(
    pool: asyncpg.Pool,
    mode: str,
    instance: str | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    try:
        async with pool.acquire() as conn:
            raw = await conn.fetchval(
                """
                SELECT record_worker_start_and_check_storm(
                    $1, $2, $3::jsonb, $4, $5, $6
                )
                """,
                mode,
                instance,
                _json_payload(metadata),
                WORKER_STORM_MAX_STARTS,
                WORKER_STORM_WINDOW_SECONDS,
                WORKER_STORM_BACKOFF_CAP_SECONDS,
            )
        result = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except Exception:
        logger.debug("worker start storm check unavailable", exc_info=True)
        return {"storm": False}

    backoff_s = float(result.get("backoff_seconds") or 0)
    if backoff_s > 0:
        logger.warning(
            "Worker start storm detected for %s/%s: %s starts in %ss; backing off %.1fs",
            mode,
            instance or "default",
            result.get("count"),
            result.get("window_seconds"),
            backoff_s,
        )
        await asyncio.sleep(backoff_s)
    return result


async def _register_worker_instance(
    pool: asyncpg.Pool,
    mode: str,
    instance: str | None,
) -> str | None:
    metadata = _worker_metadata()
    try:
        await _recover_worker_runtime(pool)
        storm = await _record_worker_start_and_maybe_backoff(
            pool, mode, instance, metadata
        )
        if isinstance(storm, dict):
            metadata["start_storm"] = storm
        async with pool.acquire() as conn:
            worker_id = await conn.fetchval(
                "SELECT register_worker_instance($1, $2, $3::jsonb)",
                mode,
                instance,
                _json_payload(metadata),
            )
        return str(worker_id) if worker_id else None
    except Exception:
        logger.warning("worker runtime registration failed", exc_info=True)
        return None


async def _mark_worker_seen(
    pool: asyncpg.Pool | None,
    worker_id: str | None,
    *,
    status: str = "running",
) -> None:
    if not pool or not worker_id:
        return
    try:
        async with pool.acquire() as conn:
            await conn.fetchval(
                "SELECT mark_worker_instance_seen($1::uuid, $2)",
                worker_id,
                status,
            )
    except Exception:
        logger.debug("worker liveness update failed", exc_info=True)


async def _mark_worker_stopped(
    pool: asyncpg.Pool | None,
    worker_id: str | None,
    *,
    reason: str | None = None,
) -> None:
    if not pool or not worker_id:
        return
    try:
        async with pool.acquire() as conn:
            await conn.fetchval(
                "SELECT mark_worker_instance_stopped($1::uuid, $2)",
                worker_id,
                reason,
            )
    except Exception:
        logger.debug("worker stopped update failed", exc_info=True)


async def _start_worker_task_run(
    pool: asyncpg.Pool | None,
    worker_id: str | None,
    task_type: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> str | None:
    if not pool or not worker_id:
        return None
    try:
        async with pool.acquire() as conn:
            run_id = await conn.fetchval(
                "SELECT start_worker_task_run($1::uuid, $2, $3::jsonb)",
                worker_id,
                task_type,
                _json_payload(metadata or {}),
            )
        return str(run_id) if run_id else None
    except Exception:
        logger.debug("worker task run start failed for %s", task_type, exc_info=True)
        return None


async def _complete_worker_task_run(
    pool: asyncpg.Pool | None,
    run_id: str | None,
    result: Any,
) -> None:
    if not pool or not run_id:
        return
    try:
        async with pool.acquire() as conn:
            await conn.fetchval(
                "SELECT complete_worker_task_run($1::uuid, $2::jsonb)",
                run_id,
                _json_payload(result),
            )
    except Exception:
        logger.debug("worker task run completion failed", exc_info=True)


async def _fail_worker_task_run(
    pool: asyncpg.Pool | None,
    run_id: str | None,
    error: str,
    result: Any | None = None,
) -> None:
    if not pool or not run_id:
        return
    try:
        async with pool.acquire() as conn:
            await conn.fetchval(
                "SELECT fail_worker_task_run($1::uuid, $2, $3::jsonb)",
                run_id,
                error,
                _json_payload(result) if result is not None else None,
            )
    except Exception:
        logger.debug("worker task run failure update failed", exc_info=True)


async def _discard_worker_task_run(
    pool: asyncpg.Pool | None,
    run_id: str | None,
    result: Any,
) -> None:
    if not pool or not run_id:
        return
    try:
        async with pool.acquire() as conn:
            await conn.fetchval(
                "SELECT discard_worker_task_run($1::uuid, $2::jsonb)",
                run_id,
                _json_payload(result),
            )
    except Exception:
        logger.debug("worker task run discard failed", exc_info=True)


async def _record_worker_task_outcome(
    pool: asyncpg.Pool | None,
    worker_id: str | None,
    task_type: str,
    *,
    status: str,
    started_at: datetime,
    finished_at: datetime,
    result: Any | None = None,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str | None:
    if not pool or not worker_id:
        return None
    try:
        async with pool.acquire() as conn:
            run_id = await conn.fetchval(
                """
                SELECT record_worker_task_outcome(
                    $1::uuid,
                    $2,
                    $3,
                    $4::timestamptz,
                    $5::timestamptz,
                    $6::jsonb,
                    $7,
                    $8::jsonb
                )
                """,
                worker_id,
                task_type,
                status,
                started_at,
                finished_at,
                _json_payload(result) if result is not None else None,
                error,
                _json_payload(metadata or {}),
            )
        return str(run_id) if run_id else None
    except Exception:
        logger.debug(
            "worker task outcome recording failed for %s", task_type, exc_info=True
        )
        return None


# ---------------------------------------------------------------------------
# HeartbeatWorker — Timer that checks if heartbeat is due and submits events
# ---------------------------------------------------------------------------


class HeartbeatWorker:
    """Timer that checks if heartbeat is due and submits gateway events.

    This worker does NOT execute heartbeats — it only decides WHEN to fire
    and submits the heartbeat payload as a gateway event. The GatewayConsumer
    dequeues and executes the actual heartbeat logic.
    """

    BELIEF_UPDATES_BUFFER_SIZE = 32

    def __init__(self, instance: str | None = None):
        self.instance = instance or os.getenv("HEXIS_INSTANCE")
        self.pool: asyncpg.Pool | None = None
        self.tool_registry = None
        self.running = False
        self.worker_id: str | None = None
        self._belief_listener = None
        self._belief_updates_recent: deque[dict[str, Any]] = deque(
            maxlen=self.BELIEF_UPDATES_BUFFER_SIZE
        )

    async def connect(self) -> None:
        self.pool = await asyncpg.create_pool(
            dsn=db_dsn_from_env(self.instance),
            min_size=1,
            max_size=5,
        )
        self.worker_id = await _register_worker_instance(
            self.pool, "heartbeat", self.instance
        )
        from services.belief_propagation import BeliefUpdateListener

        self._belief_listener = BeliefUpdateListener(self.pool, worker_id="heartbeat")
        self._belief_listener.add_handler(self._record_belief_update)
        await self._belief_listener.start()
        logger.info("HeartbeatWorker connected to database")

    async def disconnect(self) -> None:
        if self._belief_listener is not None:
            await self._belief_listener.stop()
            self._belief_listener = None
        if self.pool:
            await _mark_worker_stopped(self.pool, self.worker_id, reason="shutdown")
            await self.pool.close()
            logger.info("HeartbeatWorker disconnected")

    async def _record_belief_update(self, payload: dict[str, Any]) -> None:
        self._belief_updates_recent.append(payload)
        logger.info(
            "Belief update received: memory=%s kind=%s log=%s",
            payload.get("memory_id"),
            payload.get("change_kind"),
            payload.get("log_id"),
        )

    def get_recent_belief_updates(self) -> list[dict[str, Any]]:
        return list(self._belief_updates_recent)

    async def _submit_heartbeat_if_due(self) -> dict[str, Any]:
        if not self.pool:
            return {"skipped": True, "reason": "no_pool"}
        async with self.pool.acquire() as conn:
            payload = await run_heartbeat(conn)
            if not payload:
                return {"skipped": True, "reason": "not_due"}
            heartbeat_id = payload.get("heartbeat_id")
            if heartbeat_id:
                logger.info(f"Heartbeat due: {heartbeat_id} — submitting to gateway")

        run_id = await _start_worker_task_run(
            self.pool,
            self.worker_id,
            "heartbeat",
            metadata={"heartbeat_id": str(heartbeat_id) if heartbeat_id else None},
        )
        # Submit the full payload as a gateway event for the consumer
        try:
            gw = Gateway(self.pool)
            await gw.submit(
                EventSource.HEARTBEAT,
                f"heartbeat:{heartbeat_id or 'unknown'}",
                payload,
            )
            result = {
                "heartbeat_id": str(heartbeat_id) if heartbeat_id else None,
                "submitted": True,
            }
            await _complete_worker_task_run(self.pool, run_id, result)
            return result
        except Exception:
            logger.error("Failed to submit heartbeat event", exc_info=True)
            await _fail_worker_task_run(
                self.pool,
                run_id,
                "failed to submit heartbeat event",
                {"heartbeat_id": str(heartbeat_id) if heartbeat_id else None},
            )
            return {
                "failed": True,
                "heartbeat_id": str(heartbeat_id) if heartbeat_id else None,
            }

    async def run(self) -> None:
        self.running = True
        logger.info("HeartbeatWorker (timer) starting...")
        await self.connect()

        try:
            while self.running:
                try:
                    await _mark_worker_seen(self.pool, self.worker_id)
                    await self._probe_capabilities_if_due()
                    if await self._is_agent_terminated():
                        logger.info("Agent is terminated; heartbeat timer exiting.")
                        break
                    if not await self._is_agent_ready():
                        await asyncio.sleep(POLL_INTERVAL)
                        continue
                    if (
                        not await self._is_active_hour()
                        and not await self._has_pending_inbound_wake()
                    ):
                        logger.debug("Outside active hours; skipping heartbeat.")
                        await asyncio.sleep(POLL_INTERVAL * 10)
                        continue
                    await self._submit_heartbeat_if_due()
                except Exception as exc:
                    logger.error(f"Heartbeat timer error: {exc}")
                await asyncio.sleep(POLL_INTERVAL)
        finally:
            await self.disconnect()

    def stop(self) -> None:
        self.running = False
        logger.info("HeartbeatWorker (timer) stopping...")

    async def _probe_capabilities_if_due(self) -> dict[str, Any]:
        if not self.pool or self.tool_registry is None:
            return {"skipped": True, "reason": "registry_not_ready"}
        from services.capability_probe import run_probe_if_due

        results = await run_probe_if_due(
            self.pool,
            self.tool_registry,
            worker_name="heartbeat",
            worker_id=self.worker_id,
        )
        if results is None:
            return {"skipped": True, "reason": "not_due"}
        return {"measured": len(results)}

    async def _is_agent_terminated(self) -> bool:
        if not self.pool:
            return False
        try:
            async with self.pool.acquire() as conn:
                return await is_agent_terminated(conn)
        except Exception:
            return False

    async def _is_agent_ready(self) -> bool:
        if not self.pool:
            return False
        try:
            async with self.pool.acquire() as conn:
                return bool(
                    await conn.fetchval(
                        "SELECT is_agent_configured() AND is_init_complete()"
                    )
                )
        except Exception:
            return False

    async def _is_active_hour(self) -> bool:
        """Check if the current time is within configured active hours (DB-owned)."""
        if not self.pool:
            return True
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchval("SELECT is_within_active_hours()")
            return bool(result) if result is not None else True
        except Exception:
            return True

    async def _has_pending_inbound_wake(self) -> bool:
        if not self.pool:
            return False
        async with self.pool.acquire() as conn:
            return await has_pending_inbound_disposition_wake(conn)


# ---------------------------------------------------------------------------
# Heartbeat event handler — registered with GatewayConsumer
# ---------------------------------------------------------------------------


def _extract_heartbeat_context(payload: dict) -> dict:
    """Extract heartbeat context from the run_heartbeat() payload."""
    external_calls = payload.get("external_calls")
    if not isinstance(external_calls, list):
        return payload

    for call in external_calls:
        if not isinstance(call, dict):
            continue
        call_type = str(call.get("call_type") or "")
        if call_type == "think":
            call_input = call.get("input") or {}
            if isinstance(call_input, dict):
                context = call_input.get("context")
                if isinstance(context, dict):
                    return context
                return call_input

    return payload


async def _is_agentic_heartbeat_enabled(conn) -> bool:
    """Check if the agentic heartbeat loop is enabled via config."""
    try:
        val = await conn.fetchval("SELECT get_config('heartbeat.use_agentic_loop')")
        if val is None:
            return False
        if isinstance(val, str):
            return val.strip().lower() in ("true", "1", "yes", "on")
        return bool(val)
    except Exception:
        return False


def create_heartbeat_handler(
    *,
    pool: asyncpg.Pool,
    bridge: RabbitMQBridge | None,
    tool_registry,
    call_processor: ExternalCallProcessor,
    stop_callback: Any = None,
):
    """Factory that returns a heartbeat event handler for the GatewayConsumer.

    The handler receives a GatewayEvent whose payload is the full run_heartbeat()
    output and executes the heartbeat (agentic or legacy path).
    """

    async def _publish_outbox(messages: list[dict]) -> None:
        if not messages:
            return
        async with pool.acquire() as conn:
            # Deliver to the dashboard inbox first — heartbeat messages must
            # not depend on the profile-gated channel worker to be seen (#107).
            try:
                await tee_outbox_to_web_inbox(conn, messages)
            except Exception:
                logger.warning(
                    "web_inbox tee failed for heartbeat outbox", exc_info=True
                )
            if bridge and await channel_adapters_active(conn):
                await bridge.publish_outbox_payloads(messages)

    async def handle_heartbeat(event: GatewayEvent) -> dict[str, Any] | None:
        payload = event.payload
        heartbeat_id = payload.get("heartbeat_id")
        if heartbeat_id:
            logger.info(f"Consumer executing heartbeat: {heartbeat_id}")

        # Publish outbox messages from initialization
        outbox_messages = payload.get("outbox_messages")
        if isinstance(outbox_messages, list):
            await _publish_outbox(outbox_messages)

        async with pool.acquire() as conn:
            # Agentic heartbeat path
            if await _is_agentic_heartbeat_enabled(conn) and tool_registry:
                if not heartbeat_id:
                    logger.warning("Agentic heartbeat: no heartbeat_id in payload")
                    return {"path": "agentic", "note": "no heartbeat_id"}

                context = _extract_heartbeat_context(payload)

                try:
                    result = await run_agentic_heartbeat(
                        conn,
                        pool=pool,
                        registry=tool_registry,
                        heartbeat_id=str(heartbeat_id),
                        context=context,
                    )
                    logger.info(
                        "Agentic heartbeat %s completed: %d tools, %d energy, reason=%s",
                        str(heartbeat_id)[:8],
                        len(result.get("tool_calls_made", [])),
                        result.get("energy_spent", 0),
                        result.get("stopped_reason", "?"),
                    )
                except Exception as exc:
                    logger.error(f"Agentic heartbeat failed: {exc}")
                    result = {
                        "text": f"Heartbeat failed: {exc}",
                        "tool_calls_made": [],
                        "energy_spent": 0,
                        "stopped_reason": "error",
                    }

                try:
                    fin = await finalize_heartbeat(
                        conn,
                        heartbeat_id=str(heartbeat_id),
                        result=result,
                    )
                    outbox_messages = fin.get("outbox_messages")
                    if isinstance(outbox_messages, list):
                        await _publish_outbox(outbox_messages)
                except Exception as exc:
                    logger.error(f"Heartbeat finalization failed: {exc}")

                return {
                    "path": "agentic",
                    "stopped_reason": result.get("stopped_reason"),
                }

            # Legacy heartbeat path
            external_calls = payload.get("external_calls")
            if not isinstance(external_calls, list):
                return {"path": "legacy", "note": "no external_calls"}

            for call in external_calls:
                if not isinstance(call, dict):
                    continue
                call_type = str(call.get("call_type") or "")
                call_input = call.get("input") or {}
                if not isinstance(call_input, dict):
                    call_input = {}
                try:
                    result = await call_processor.process_call_payload(
                        conn, call_type, call_input
                    )
                    applied = await call_processor.apply_result(conn, call, result)
                except Exception as exc:
                    logger.error(f"Error processing external call: {exc}")
                    continue

                if isinstance(applied, dict):
                    outbox_messages = applied.get("outbox_messages")
                    if isinstance(outbox_messages, list):
                        await _publish_outbox(outbox_messages)

                if (
                    isinstance(result, dict)
                    and result.get("kind") == "heartbeat_decision"
                    and "decision" in result
                    and heartbeat_id
                ):
                    try:
                        exec_result = await execute_heartbeat_decision(
                            conn,
                            heartbeat_id=str(heartbeat_id),
                            decision=result["decision"],
                            call_processor=call_processor,
                            pre_executed_actions=result.get("rlm_repl_actions"),
                        )
                    except Exception:
                        await conn.fetchval(
                            "SELECT release_active_heartbeat($1)", str(heartbeat_id)
                        )
                        raise
                    if isinstance(exec_result, dict):
                        outbox_messages = exec_result.get("outbox_messages")
                        if isinstance(outbox_messages, list):
                            await _publish_outbox(outbox_messages)
                        if exec_result.get("terminated") is True:
                            logger.info("Termination executed; stopping workers.")
                            if stop_callback:
                                stop_callback()
                    return {"path": "legacy"}

            return {"path": "legacy"}

    return handle_heartbeat


# ---------------------------------------------------------------------------
# MaintenanceWorker — runs directly (atomic check+execute tasks)
# ---------------------------------------------------------------------------


class MaintenanceWorker:
    """Subconscious maintenance loop: consolidates/prunes substrate on its own trigger."""

    def __init__(self, instance: str | None = None):
        self.instance = instance or os.getenv("HEXIS_INSTANCE")
        self.pool: asyncpg.Pool | None = None
        self.running = False
        self.bridge: RabbitMQBridge | None = None
        self.tool_registry = None
        self.worker_id: str | None = None

    async def connect(self) -> None:
        self.pool = await asyncpg.create_pool(
            dsn=db_dsn_from_env(self.instance), min_size=1, max_size=5
        )
        self.worker_id = await _register_worker_instance(
            self.pool, "maintenance", self.instance
        )
        logger.info("Connected to database")
        self.bridge = RabbitMQBridge(self.pool)
        await self.bridge.ensure_ready()

    async def disconnect(self) -> None:
        if self.pool:
            await _mark_worker_stopped(self.pool, self.worker_id, reason="shutdown")
            await self.pool.close()
            logger.info("Disconnected from database")

    async def _run_observed_task(
        self,
        task_type: str,
        runner: Callable[[], Awaitable[Any]],
    ) -> Any:
        started_at = datetime.now(timezone.utc)
        try:
            result = await runner()
        except Exception as exc:
            await _record_worker_task_outcome(
                self.pool,
                self.worker_id,
                task_type,
                status="failed",
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                result={"task_type": task_type},
                error=str(exc),
            )
            logger.exception("Maintenance task %s failed", task_type)
            return {"failed": True, "error": str(exc)}

        if isinstance(result, dict) and result.get("failed"):
            await _record_worker_task_outcome(
                self.pool,
                self.worker_id,
                task_type,
                status="failed",
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                result=result,
                error=str(result.get("error") or "task reported failure"),
            )
        elif _result_has_work(result):
            await _record_worker_task_outcome(
                self.pool,
                self.worker_id,
                task_type,
                status="completed",
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                result=result,
            )
        return result

    def _maintenance_task_runners(
        self,
    ) -> list[tuple[str, Callable[[], Awaitable[Any]]]]:
        return [
            ("capability_probe", self._probe_capabilities_if_due),
            ("automation_suggestions", self._run_automation_suggestions),
            ("inbox_poll", self._run_inbox_poll),
            ("outbox_delivery", self._run_outbox_delivery),
            ("scheduled_tasks", self._run_scheduled_tasks),
            ("ambient_responsibilities", self._run_ambient_responsibilities),
            ("hmx_reembedding", self._run_hmx_reembedding),
            ("subconscious_maintenance", self._run_maintenance_if_due),
            ("subconscious_decider", self._run_subconscious_if_due),
            ("reconsolidation", self._run_reconsolidation_if_pending),
            ("memory_embedding", self._run_memory_embedding),
            ("contradiction_detection", self._run_contradiction_detection),
            ("contradiction_digest", self._run_contradiction_digest),
            ("recmem", self._run_recmem_if_enabled),
            ("source_chunk_embedding", self._run_source_chunk_embedding),
            ("memory_summarization", self._run_memory_rest_if_enabled),
            ("retention_surface", self._run_retention_surface),
            ("conscious_extraction", self._run_extraction_if_enabled),
            ("origin_seed", self._run_origin_seed_if_enabled),
            ("skill_improvement", self._run_skill_improvement_if_due),
            ("learning_skill_application", self._run_approved_learning_skill),
            ("gmail_backfill", self._run_gmail_backfill_jobs),
            ("channel_backfill", self._run_channel_backfill_jobs),
            ("connector_cognition", self._run_connector_cognition),
            ("belief_update_retention", self._prune_belief_update_log),
            ("ingestion_jobs", self._run_ingestion_jobs),
        ]

    async def _prune_belief_update_log(self) -> dict[str, Any]:
        if not self.pool:
            return {"skipped": True, "reason": "no_pool"}
        async with self.pool.acquire() as conn:
            deleted = int(await conn.fetchval("SELECT prune_belief_update_log()") or 0)
        if deleted:
            logger.info("Pruned %s stale belief-update events", deleted)
            return {"deleted": deleted}
        return {"skipped": True, "reason": "nothing_expired"}

    async def _probe_capabilities_if_due(self) -> dict[str, Any]:
        if not self.pool or self.tool_registry is None:
            return {"skipped": True, "reason": "registry_not_ready"}
        from services.capability_probe import run_probe_if_due

        results = await run_probe_if_due(
            self.pool,
            self.tool_registry,
            worker_name="maintenance",
            worker_id=self.worker_id,
        )
        if results is None:
            return {"skipped": True, "reason": "not_due"}
        return {"measured": len(results)}

    async def _run_automation_suggestions(self) -> dict[str, Any]:
        if not self.pool:
            return {"skipped": True, "reason": "no_pool"}
        from services.automation_suggestions import refresh_automation_suggestions

        result = await refresh_automation_suggestions(self.pool)
        if not result.get("skipped"):
            logger.info("Automation suggestion refresh: %s", result)
        return result

    async def _publish_outbox(self, messages: list[dict]) -> None:
        if not messages:
            return
        if self.bridge:
            await self.bridge.publish_outbox_payloads(messages)

    async def _tee_outbox_to_web_inbox(
        self, conn: asyncpg.Connection, messages: list[dict]
    ) -> int:
        return await tee_outbox_to_web_inbox(conn, messages)

    async def _run_inbox_poll(self) -> dict[str, Any]:
        if not self.bridge:
            return {"skipped": True, "reason": "no_bridge"}
        await self.bridge.poll_inbox_messages()
        return {"skipped": True, "reason": "poll_only"}

    async def _run_outbox_delivery(self) -> dict[str, Any]:
        """Drain the DB-native outbox (tool-queued user messages) to RabbitMQ.

        publish_outbox_payloads publishes in order and returns the count that
        succeeded (stopping at the first failure), so mark that prefix published
        and requeue the rest for the next tick.
        """
        if not self.pool or not self.bridge:
            return {"skipped": True, "reason": "no_pool_or_bridge"}
        async with self.pool.acquire() as conn:
            raw = await conn.fetchval("SELECT claim_pending_outbox($1::int)", 50)
            claimed = json.loads(raw) if isinstance(raw, str) else (raw or [])
            if not claimed:
                return {"skipped": True, "reason": "no_pending_outbox"}
            ids = [c["id"] for c in claimed]
            envelopes = [c["envelope"] for c in claimed]
            # Tee to the dashboard inbox BEFORE the RabbitMQ hop: user-bound
            # messages must not depend on the profile-gated channel worker to
            # become visible (#107). Idempotent per envelope id, so the channel
            # worker teeing the same message later is a no-op.
            teed = await self._tee_outbox_to_web_inbox(conn, envelopes)
            if await channel_adapters_active(conn):
                published = await self.bridge.publish_outbox_payloads(envelopes)
                broker = "published"
            else:
                # No adapter exists to consume the queue; the web_inbox tee
                # above is the delivery. Skip the broker so messages don't
                # accumulate unconsumed (#107).
                published = len(ids)
                broker = "skipped_no_adapters"
            if published > 0:
                await conn.fetchval(
                    "SELECT mark_outbox_published($1::uuid[])", ids[:published]
                )
            if published < len(ids):
                await conn.fetchval(
                    "SELECT requeue_outbox($1::uuid[])", ids[published:]
                )
            return {
                "claimed": len(ids),
                "published": int(published),
                "requeued": max(len(ids) - int(published), 0),
                "web_inbox": int(teed),
                "broker": broker,
            }

    async def _run_scheduled_tasks(self) -> dict[str, Any]:
        if not self.pool:
            return {"skipped": True, "reason": "no_pool"}
        async with self.pool.acquire() as conn:
            payload = await run_scheduled_tasks(conn)
            if not isinstance(payload, dict):
                return {"skipped": True, "reason": "no_scheduled_payload"}
            outbox_messages = payload.get("outbox_messages")
            if isinstance(outbox_messages, list):
                delivered = await self._tee_outbox_to_web_inbox(conn, outbox_messages)
                if delivered:
                    payload["web_inbox_delivered"] = delivered
                await self._publish_outbox(outbox_messages)
            executed = (
                payload.get("ran")
                or payload.get("executed_count")
                or payload.get("executed")
            )
            ran_tasks = payload.get("ran_tasks") or []
            if executed:
                try:
                    await Gateway(self.pool).record(
                        EventSource.CRON,
                        "cron::scheduled",
                        {"executed": executed, "tasks": ran_tasks},
                    )
                except Exception:
                    logger.debug("Gateway record failed (non-fatal)", exc_info=True)
            # Recompute next_run_at for cron-expression tasks (cron_next_fire in the DB)
            cron_task_ids = payload.get("cron_task_ids") or []
            if cron_task_ids:
                try:
                    await recompute_cron_next_runs(conn, cron_task_ids)
                except Exception:
                    logger.warning("Cron recompute failed (non-fatal)", exc_info=True)
            if executed:
                return payload
            return {
                "skipped": True,
                "reason": "no_due_scheduled_tasks",
                "payload": payload,
            }

    async def _run_ambient_responsibilities(self) -> dict[str, Any]:
        if not self.pool:
            return {"skipped": True, "reason": "no_pool"}
        payload = await run_ambient_responsibility_step(self.pool)
        if not isinstance(payload, dict):
            return {"skipped": True, "reason": "no_ambient_payload"}
        outbox_messages = payload.get("outbox_messages")
        if isinstance(outbox_messages, list) and outbox_messages:
            async with self.pool.acquire() as conn:
                delivered = await self._tee_outbox_to_web_inbox(conn, outbox_messages)
                if delivered:
                    payload["web_inbox_delivered"] = delivered
            await self._publish_outbox(outbox_messages)
        if not payload.get("skipped"):
            try:
                await Gateway(self.pool).record(
                    EventSource.CRON,
                    "cron::ambient_responsibility",
                    {
                        "summary": {
                            k: v for k, v in payload.items() if k != "outbox_messages"
                        }
                    },
                )
            except Exception:
                logger.debug("Gateway record failed (non-fatal)", exc_info=True)
        return payload

    async def _run_maintenance_if_due(self) -> dict[str, Any]:
        if not self.pool:
            return {"skipped": True, "reason": "no_pool"}
        async with self.pool.acquire() as conn:
            stats = await run_maintenance_if_due(conn, {})
            if stats is None:
                return {"skipped": True, "reason": "not_due"}
            if not stats.get("skipped"):
                logger.info(f"Subconscious maintenance: {stats}")
                try:
                    await Gateway(self.pool).record(
                        EventSource.MAINTENANCE,
                        "maintenance::consolidation",
                        {"stats": stats},
                    )
                except Exception:
                    logger.debug("Gateway record failed (non-fatal)", exc_info=True)
            return stats

    async def _run_hmx_reembedding(self) -> dict[str, Any]:
        if not self.pool:
            return {"skipped": True, "reason": "no_pool"}
        async with self.pool.acquire() as conn:
            result = await run_hmx_reembed_step(conn)
            if not result.get("skipped"):
                logger.info("HMX re-embedding step: %s", result)
            return result

    async def _run_subconscious_if_due(self) -> dict[str, Any]:
        if not self.pool:
            return {"skipped": True, "reason": "no_pool"}
        async with self.pool.acquire() as conn:
            should_run = await should_run_subconscious_decider(conn)
            if not should_run:
                return {"skipped": True, "reason": "not_due"}
            result = await run_subconscious_decider(conn)
            await mark_subconscious_decider_run(conn)
            logger.info(f"Subconscious decider: {result}")
            return result

    async def _run_reconsolidation_if_pending(self) -> dict[str, Any]:
        if not self.pool:
            return {"skipped": True, "reason": "no_pool"}
        async with self.pool.acquire() as conn:
            has_pending = await conn.fetchval("SELECT has_pending_reconsolidation()")
            if not has_pending:
                return {"skipped": True, "reason": "no_pending_tasks"}
            result = await run_reconsolidation_step(conn)
            if not result.get("skipped"):
                logger.info(f"Reconsolidation step: {result}")
            return result

    async def _run_contradiction_detection(self) -> dict[str, Any]:
        if not self.pool:
            return {"skipped": True, "reason": "no_pool"}
        from services.contradictions import run_contradiction_detection_step

        async with self.pool.acquire() as conn:
            result = await run_contradiction_detection_step(conn)
        if not result.get("skipped"):
            logger.info("Contradiction detection: %s", result)
        return result

    async def _run_contradiction_digest(self) -> dict[str, Any]:
        if not self.pool:
            return {"skipped": True, "reason": "no_pool"}
        from services.contradictions import publish_contradiction_digest

        async with self.pool.acquire() as conn:
            result = await publish_contradiction_digest(conn)
        if not result.get("skipped"):
            logger.info("Contradiction review digest: %s", result)
        return result

    async def _run_ingestion_jobs(self) -> dict[str, Any]:
        """Durable ingestion jobs (#87): the queue table is the state — no
        state-doc gate, matching outbox delivery."""
        if not self.pool:
            return {"skipped": True, "reason": "no_pool"}
        from services.ingest.jobs import run_ingestion_jobs_step

        handled = await run_ingestion_jobs_step(self.pool)
        if handled:
            return {"handled": handled}
        return {"skipped": True, "reason": "no_due_ingestion_jobs"}

    async def _run_gmail_backfill_jobs(self) -> dict[str, Any]:
        if not self.pool:
            return {"skipped": True, "reason": "no_pool"}
        handled = await run_gmail_backfill_step(self.pool)
        if handled:
            logger.info("Gmail connector backfill jobs handled: %s", handled)
            return {"handled": handled}
        return {"skipped": True, "reason": "no_due_gmail_backfill_jobs"}

    async def _run_channel_backfill_jobs(self) -> dict[str, Any]:
        if not self.pool:
            return {"skipped": True, "reason": "no_pool"}
        handled = await run_channel_backfill_step(self.pool)
        if handled:
            logger.info("Channel connector backfill jobs handled: %s", handled)
            return {"handled": handled}
        return {"skipped": True, "reason": "no_due_channel_backfill_jobs"}

    async def _run_connector_cognition(self) -> dict[str, Any]:
        if not self.pool:
            return {"skipped": True, "reason": "no_pool"}
        async with self.pool.acquire() as conn:
            user_model = await run_user_model_synthesis_step(conn)
            if not user_model.get("skipped"):
                logger.info("Connector user-model synthesis: %s", user_model)
            importance = await run_connector_importance_step(conn)
            if not importance.get("skipped"):
                logger.info("Connector importance detection: %s", importance)
        if user_model.get("skipped") and importance.get("skipped"):
            return {
                "skipped": True,
                "reason": "no_connector_cognition_work",
                "user_model": user_model,
                "importance": importance,
            }
        return {"user_model": user_model, "importance": importance}

    async def _run_recmem_if_enabled(self) -> dict[str, Any]:
        if not self.pool:
            return {"skipped": True, "reason": "no_pool"}
        async with self.pool.acquire() as conn:
            summary: dict[str, Any] = {}
            did_work = False
            embed_result = await run_recmem_embed_step(conn)
            if not embed_result.get("skipped"):
                logger.info("RecMem embed step: %s", embed_result)
                did_work = True
            summary["embed"] = embed_result
            route_result = await run_recmem_route_step(conn)
            if not route_result.get("skipped"):
                logger.info("RecMem route step: %s", route_result)
                did_work = True
            summary["route"] = route_result
            should_sweep = bool(await conn.fetchval("SELECT should_run_recmem_sweep()"))
            if should_sweep:
                sweep_result = await run_recmem_sweep_step(conn)
                await conn.fetchval(
                    "SELECT mark_recmem_sweep_run($1::jsonb)", json.dumps(sweep_result)
                )
                if sweep_result.get("processed", 0):
                    logger.info("RecMem sweep step: %s", sweep_result)
                summary["sweep"] = sweep_result
                did_work = True

            # Scene consolidation (#73): sessions gone quiet become one
            # episode_create task covering the whole conversation.
            if bool(await conn.fetchval("SELECT should_run_scene_consolidation()")):
                scene_result = await conn.fetchval(
                    "SELECT enqueue_scene_consolidations()"
                )
                scene_doc = (
                    json.loads(scene_result)
                    if isinstance(scene_result, str)
                    else (scene_result or {})
                )
                await conn.fetchval(
                    "SELECT mark_scene_consolidation_run($1::jsonb)",
                    json.dumps(scene_doc),
                )
                if scene_doc.get("enqueued"):
                    logger.info("Scene consolidation: %s", scene_doc)
                    did_work = True
                summary["scene_consolidation"] = scene_doc

            task_batch_size = int(
                await conn.fetchval(
                    "SELECT COALESCE(get_config_int('memory.recmem_task_batch_size'), 3)"
                )
                or 3
            )
            consolidation_results = []
            for _ in range(max(task_batch_size, 1)):
                result = await run_recmem_consolidation_step(conn)
                if result.get("skipped"):
                    break
                did_work = True
                consolidation_results.append(result)
            if consolidation_results:
                summary["consolidation"] = consolidation_results
            if not did_work:
                return {"skipped": True, "reason": "no_recmem_work", "steps": summary}
            return summary

    async def _run_source_chunk_embedding(self) -> dict[str, Any]:
        """Embed pending source-document chunks (deferred from ingestion so
        the pipeline never blocks on the embedding sidecar)."""
        if not self.pool:
            return {"skipped": True, "reason": "no_pool"}
        async with self.pool.acquire() as conn:
            result = await run_source_chunk_embed_step(conn)
        if not result.get("skipped"):
            logger.info("Source chunk embed step: %s", result)
        return result

    async def _run_memory_embedding(self) -> dict[str, Any]:
        """Embed pending durable memories off the memory-creation path."""
        if not self.pool:
            return {"skipped": True, "reason": "no_pool"}
        async with self.pool.acquire() as conn:
            result = await run_memory_embed_step(conn)
        if not result.get("skipped"):
            logger.info("Memory embed step: %s", result)
        return result

    async def _run_memory_rest_if_enabled(self) -> dict[str, Any]:
        """Drain the memory-consolidation summarization queue (LLM compaction +
        distill-upward). Consolidation/pruning themselves run in the DB maintenance
        pass; this only does the LLM step. No-op unless retention.enabled."""
        if not self.pool:
            return {"skipped": True, "reason": "no_pool"}
        async with self.pool.acquire() as conn:
            if not bool(
                await conn.fetchval(
                    "SELECT COALESCE(get_config_bool('retention.enabled'), false)"
                )
            ):
                return {"skipped": True, "reason": "retention_disabled"}
            result = await run_memory_summarization_step(conn)
            if not result.get("skipped"):
                logger.info("Memory summarization: %s", result)
            return result

    async def _run_retention_surface(self) -> dict[str, Any]:
        if not self.pool:
            return {"skipped": True, "reason": "no_pool"}
        async with self.pool.acquire() as conn:
            return await publish_retention_surface(conn)

    async def _run_extraction_if_enabled(self) -> dict[str, Any]:
        """Sweep conscious episodes (chat turns + heartbeat episodes) into
        selective durable memories (#37). No-op unless extraction.enabled."""
        if not self.pool:
            return {"skipped": True, "reason": "no_pool"}
        async with self.pool.acquire() as conn:
            result = await run_conscious_extraction_step(conn)
            if not result.get("skipped"):
                logger.info("Conscious extraction: %s", result)
            return result

    async def _run_origin_seed_if_enabled(self) -> dict[str, Any]:
        """Keep origin memories seeded (#40). Idempotent and config-gated in
        the DB, so flipping origin_memories.enabled takes effect on the next
        tick — no manual SQL, no re-consent. Advisory: a failure (e.g. the
        embedding service being down) waits for the next tick, loudly."""
        if not self.pool:
            return {"skipped": True, "reason": "no_pool"}
        async with self.pool.acquire() as conn:
            raw = await conn.fetchval("SELECT seed_origin_memories()")
        result = json.loads(raw) if isinstance(raw, str) else (raw or {})
        if result.get("seeded"):
            logger.info("Origin memories seeded: %s", result)
        if result.get("seeded"):
            return result
        return {
            "skipped": True,
            "reason": "origin_memories_already_seeded",
            "result": result,
        }

    async def _run_skill_improvement_if_due(self) -> dict[str, Any]:
        if not self.pool:
            return {"skipped": True, "reason": "no_pool"}
        async with self.pool.acquire() as conn:
            result = await run_skill_improvement_review_step(
                conn, registry=self.tool_registry
            )
            if not result.get("skipped"):
                logger.info("Skill-improvement review: %s", result)
            return result

    async def _run_approved_learning_skill(self) -> dict[str, Any]:
        if not self.pool:
            return {"skipped": True, "reason": "no_pool"}
        return await apply_approved_learning_skill_step(
            self.pool,
            registry=self.tool_registry,
        )

    async def run(self) -> None:
        self.running = True
        logger.info("Maintenance worker starting...")
        await self.connect()
        try:
            while self.running:
                try:
                    await _mark_worker_seen(self.pool, self.worker_id)
                    if await self._is_agent_terminated():
                        logger.info("Agent is terminated; maintenance worker exiting.")
                        break
                    if not await self._is_agent_ready():
                        await asyncio.sleep(POLL_INTERVAL)
                        continue
                    for task_type, runner in self._maintenance_task_runners():
                        await self._run_observed_task(task_type, runner)
                except Exception as exc:
                    logger.error(f"Maintenance loop error: {exc}")
                await asyncio.sleep(POLL_INTERVAL)
        finally:
            await self.disconnect()

    def stop(self) -> None:
        self.running = False
        logger.info("Maintenance worker stopping...")

    async def _is_agent_terminated(self) -> bool:
        if not self.pool:
            return False
        try:
            async with self.pool.acquire() as conn:
                return await is_agent_terminated(conn)
        except Exception:
            return False

    async def _is_agent_ready(self) -> bool:
        if not self.pool:
            return False
        try:
            async with self.pool.acquire() as conn:
                return bool(
                    await conn.fetchval(
                        "SELECT is_agent_configured() AND is_init_complete()"
                    )
                )
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Webhook event handler — registered with GatewayConsumer
# ---------------------------------------------------------------------------


def create_webhook_handler(*, pool: asyncpg.Pool):
    """Factory that returns a webhook event handler for the GatewayConsumer.

    Webhook events are recorded as episodic memories so the agent
    is aware that an external system sent a notification.
    """

    async def handle_webhook(event: GatewayEvent) -> dict[str, Any] | None:
        payload = event.payload
        source_name = event.session_key.removeprefix("webhook:")
        logger.info("Processing webhook event: %s (id=%d)", source_name, event.id)

        # Record the webhook as an episodic memory
        try:
            async with pool.acquire() as conn:
                summary = json.dumps(payload)[:500] if payload else "{}"
                await conn.fetchval(
                    """
                    SELECT create_episodic_memory(
                        p_content := $1,
                        p_importance := 0.4,
                        p_emotional_valence := 0.0,
                        p_context := $2::jsonb,
                        p_source_attribution := $3::jsonb,
                        p_trust_level := 0.7
                    )
                    """,
                    f"Received webhook from {source_name}: {summary}",
                    json.dumps({"type": "webhook", "source": source_name}),
                    json.dumps(
                        {
                            "kind": "webhook",
                            "ref": str(event.correlation_id),
                            "label": f"webhook:{source_name}",
                            "trust": 0.7,
                        }
                    ),
                )
        except Exception as exc:
            logger.warning("Failed to record webhook memory: %s", exc)

        return {"source": source_name, "recorded": True}

    return handle_webhook


# ---------------------------------------------------------------------------
# Main entrypoint — wires timer + consumer + maintenance
# ---------------------------------------------------------------------------


async def _amain(mode: str, instance: str | None = None) -> None:
    hb_timer = HeartbeatWorker(instance)
    maint_worker = MaintenanceWorker(instance)

    # Shared resources for the heartbeat consumer
    dsn = db_dsn_from_env(instance)
    # Bring the schema up to date before doing anything (advisory-locked, idempotent,
    # no-op if already current). Never wipes data.
    try:
        from core.agent_api import apply_migrations

        applied = await apply_migrations(dsn)
        if applied:
            logger.info(
                "applied %d schema migration(s) on startup: %s", len(applied), applied
            )
    except Exception as exc:
        logger.warning("startup migration check failed (continuing): %s", exc)
    consumer_pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)
    from core.usage import set_usage_pool

    set_usage_pool(consumer_pool)
    try:
        from core.agent_api import record_build_change

        async with consumer_pool.acquire() as conn:
            await record_build_change(conn, "worker")
    except Exception:
        logger.debug("build-change journaling failed", exc_info=True)
    bridge = RabbitMQBridge(consumer_pool)
    await bridge.ensure_ready()

    # Initialize tool registry for heartbeat execution
    tool_registry = None
    mcp_manager = None
    call_processor = ExternalCallProcessor(max_retries=MAX_RETRIES)
    try:
        from core.tools import create_full_registry, create_mcp_manager

        tool_registry = await create_full_registry(consumer_pool)
        hb_timer.tool_registry = tool_registry
        maint_worker.tool_registry = tool_registry
        call_processor.set_tool_registry(tool_registry)
        logger.info("Tool registry initialized for consumer")

        # Skill-gated MCP (#41, default): servers connect lazily when a skill
        # binding them is activated, so nothing eager-loads here. The legacy
        # eager mode remains behind mcp.skill_gated=false.
        async with consumer_pool.acquire() as conn:
            skill_gated = bool(
                await conn.fetchval(
                    "SELECT COALESCE(get_config_bool('mcp.skill_gated'), TRUE)"
                )
            )
        if skill_gated:
            logger.info("MCP is skill-gated; servers connect on skill activation")
        else:
            mcp_manager = await create_mcp_manager(tool_registry)
            mcp_count = len(mcp_manager.list_servers())
            if mcp_count > 0:
                logger.info(f"Loaded {mcp_count} MCP server(s)")
    except Exception as e:
        logger.warning(f"Failed to initialize tool registry: {e}")

    # Build the gateway consumer
    consumer = GatewayConsumer(consumer_pool, poll_interval=POLL_INTERVAL)

    def _stop_all():
        hb_timer.stop()
        maint_worker.stop()
        consumer.stop()

    heartbeat_handler = create_heartbeat_handler(
        pool=consumer_pool,
        bridge=bridge,
        tool_registry=tool_registry,
        call_processor=call_processor,
        stop_callback=_stop_all,
    )
    consumer.register(EventSource.HEARTBEAT, heartbeat_handler)
    consumer.register(EventSource.WEBHOOK, create_webhook_handler(pool=consumer_pool))

    import signal

    def shutdown(signum, frame):
        _stop_all()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    mode = (mode or "both").strip().lower()
    instance_info = f" (instance: {instance})" if instance else ""
    logger.info(f"Starting worker in {mode} mode{instance_info}")

    try:
        if mode == "heartbeat":
            await asyncio.gather(hb_timer.run(), consumer.run())
        elif mode == "maintenance":
            await maint_worker.run()
        elif mode == "both":
            await asyncio.gather(hb_timer.run(), consumer.run(), maint_worker.run())
        else:
            raise ValueError("mode must be one of: heartbeat, maintenance, both")
    finally:
        # Cleanup consumer resources
        if mcp_manager:
            try:
                await mcp_manager.shutdown()
            except Exception:
                pass
        try:
            from core.tools.mcp_runtime import MCPRuntime

            await MCPRuntime.instance().shutdown()
        except Exception:
            pass
        await consumer_pool.close()


def main() -> int:
    p = argparse.ArgumentParser(
        prog="hexis-worker", description="Run Hexis background workers."
    )
    p.add_argument(
        "--mode",
        choices=["heartbeat", "maintenance", "both"],
        default=os.getenv("HEXIS_WORKER_MODE", "both"),
        help="Which worker to run.",
    )
    p.add_argument(
        "--instance",
        "-i",
        default=os.getenv("HEXIS_INSTANCE"),
        help="Target a specific instance (overrides HEXIS_INSTANCE env var).",
    )
    args = p.parse_args()
    asyncio.run(_amain(args.mode, args.instance))
    return 0


__all__ = [
    "HeartbeatWorker",
    "MaintenanceWorker",
    "GatewayConsumer",
    "create_heartbeat_handler",
    "create_webhook_handler",
    "main",
    "MAX_RETRIES",
    "_result_has_work",
]
