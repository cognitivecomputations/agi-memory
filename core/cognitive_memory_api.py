"""
Cognitive Memory API

Thin async client for the Postgres-backed cognitive memory system.

Design:
- The database owns state and behavior (functions/views in db/*.sql).
- This module is a convenience layer for application integration.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Iterable
from uuid import UUID

import asyncpg

logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    STRATEGIC = "strategic"
    WORLDVIEW = "worldview"  # Phase 5 (ReduceScopeCreep): beliefs/values/boundaries
    GOAL = "goal"  # Phase 6 (ReduceScopeCreep): goals/intentions as memories


class GoalPriority(str, Enum):
    ACTIVE = "active"
    QUEUED = "queued"
    BACKBURNER = "backburner"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class GoalSource(str, Enum):
    CURIOSITY = "curiosity"
    USER_REQUEST = "user_request"
    IDENTITY = "identity"
    DERIVED = "derived"
    EXTERNAL = "external"


class RelationshipType(str, Enum):
    TEMPORAL_NEXT = "TEMPORAL_NEXT"
    CAUSES = "CAUSES"
    DERIVED_FROM = "DERIVED_FROM"
    CONTRADICTS = "CONTRADICTS"
    SUPPORTS = "SUPPORTS"
    INSTANCE_OF = "INSTANCE_OF"
    PARENT_OF = "PARENT_OF"
    ASSOCIATED = "ASSOCIATED"
    CONTESTED_BECAUSE = "CONTESTED_BECAUSE"


@dataclass(frozen=True)
class Memory:
    id: UUID
    type: MemoryType
    content: str
    importance: float
    relevance_score: float | None = None
    similarity: float | None = None
    source: str | None = None  # retrieval source: 'vector', 'association', 'temporal'
    trust_level: float | None = None  # epistemic trust [0..1] (DB-computed)
    source_attribution: dict[str, Any] | None = (
        None  # primary provenance (DB-stored JSON)
    )
    created_at: datetime | None = None
    emotional_valence: float | None = None
    tier: str | None = None
    source_unit_ids: list[UUID] | None = None
    valid_until: datetime | None = None
    # Compression-native substrate: how vividly the memory is currently held.
    # strength = recency/reinforcement/decay (varies now); fidelity = how lossy a
    # gist it is (1.0 until consolidation exists). Drives graded recall rendering.
    strength: float | None = None
    fidelity: float | None = None
    # SIGNED felt emotional intensity: >0 = warm/positive, <0 = painful/negative,
    # magnitude = how vivid the feeling is NOW (embered / healed / re-kindled).
    emotional_intensity: float | None = None
    # Belief confidence [0..1] from metadata (semantic beliefs); rendered so the
    # conscious mind can see how eroded or corroborated a belief currently is.
    confidence: float | None = None


@dataclass(frozen=True)
class PartialActivation:
    cluster_id: UUID
    cluster_name: str
    keywords: list[str]
    emotional_signature: dict[str, Any] | None
    cluster_similarity: float
    best_memory_similarity: float


@dataclass(frozen=True)
class RecallResult:
    memories: list[Memory]
    partial_activations: list[PartialActivation]
    query: str


@dataclass(frozen=True)
class HistorySearchResult:
    source_kind: str
    item_id: UUID
    session_id: UUID | None
    content: str
    user_text: str | None
    assistant_text: str | None
    memory_type: MemoryType | None
    occurred_at: datetime
    rank: float
    source_unit_ids: list[UUID]
    source_attribution: dict[str, Any]
    metadata: dict[str, Any]


# Default edge types for the chat reasoning subgraph: the *semantic* relations
# that carry reasoning structure. Excludes hub-forming structural/temporal edges
# (IN_EPISODE, MEMBER_OF, CLUSTER_*, goal-tree edges) that would over-connect the
# view — every co-occurring memory linking through one episode/cluster node. The
# substrate still stores those; the explore_subgraph tool can request any type.
REASONING_EDGE_TYPES = [
    "CAUSES",
    "SUPPORTS",
    "CONTRADICTS",
    "DERIVED_FROM",
    "CONTESTED_BECAUSE",
    "INSTANCE_OF",
    "ASSOCIATED",
    "TEMPORAL_NEXT",
]


@dataclass(frozen=True)
class HydratedContext:
    memories: list[Memory]
    partial_activations: list[PartialActivation]
    identity: list[dict[str, Any]]
    worldview: list[dict[str, Any]]
    emotional_state: dict[str, Any] | None
    goals: dict[str, Any] | None
    urgent_drives: list[dict[str, Any]]
    # Dynamic sub-knowledge-graph seeded from the recalled memories: the typed
    # {nodes, edges} structure (supports/contradicts/causes/...) among + around
    # them. None when disabled or nothing was recalled.
    subgraph: dict[str, Any] | None = None
    # Focused multi-hop causal/contradiction/support paths among the recalled
    # memories. The DB builds these; renderers only carry and display them.
    context_paths: dict[str, Any] | None = None
    user_model: list[dict[str, Any]] = field(default_factory=list)


def _deduplicate_memories(memories: Iterable[Memory]) -> list[Memory]:
    """Keep the highest-ranked occurrence of the same memory or exact content."""
    deduplicated: list[Memory] = []
    seen_ids: set[UUID] = set()
    seen_content: set[tuple[str, str]] = set()
    for memory in memories:
        normalized = " ".join(memory.content.split()).casefold()
        content_key = (memory.type.value, normalized)
        if memory.id in seen_ids or content_key in seen_content:
            continue
        seen_ids.add(memory.id)
        seen_content.add(content_key)
        deduplicated.append(memory)
    return deduplicated


@dataclass(frozen=True)
class MemoryInput:
    content: str
    type: MemoryType = MemoryType.EPISODIC
    importance: float = 0.5
    emotional_valence: float = 0.0
    context: dict[str, Any] | None = None
    concepts: list[str] | None = None
    source_attribution: dict[str, Any] | None = None
    source_references: Any | None = (
        None  # JSONB for semantic memories (dict or list[dict])
    )
    trust_level: float | None = None


@dataclass(frozen=True)
class RelationshipInput:
    from_id: UUID
    to_id: UUID
    relationship_type: RelationshipType
    confidence: float = 0.8
    context: str | None = None


async def _init_connection(conn: asyncpg.Connection) -> None:
    try:
        await conn.execute("LOAD 'age';")
    except Exception:
        pass
    try:
        # public FIRST (migration 0043): Hexis functions live in public;
        # ag_catalog stays on the path for AGE's cypher/agtype.
        await conn.execute("SET search_path = public, ag_catalog;")
    except Exception:
        pass


def _to_jsonb_arg(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, (dict, list)):
        import json

        return json.dumps(val)
    return val


def _uuid_text_or_none(val: UUID | str | None) -> str | None:
    if val is None:
        return None
    try:
        return str(UUID(str(val)))
    except Exception:
        return None


def _cypher_escape(value: str) -> str:
    return value.replace("'", "''")


class CognitiveMemory:
    """
    Async client for the cognitive memory database.

    Two common flows:
    - RAG hydration: `hydrate()`
    - Agent operations: `recall()`, `remember()`, `connect_memories()`
    """

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    @classmethod
    @asynccontextmanager
    async def connect(
        cls,
        dsn: str,
        **pool_kwargs: Any,
    ) -> AsyncIterator["CognitiveMemory"]:
        """
        Async context manager that owns the underlying pool.

        Usage:
            async with CognitiveMemory.connect(dsn) as mem:
                ctx = await mem.hydrate("...")
        """
        pool = await asyncpg.create_pool(dsn, init=_init_connection, **pool_kwargs)
        client = cls(pool)
        try:
            yield client
        finally:
            await pool.close()

    @classmethod
    async def create(cls, dsn: str, **pool_kwargs: Any) -> "CognitiveMemory":
        """Create a pool and return a client; call `close()` when done."""
        pool = await asyncpg.create_pool(dsn, init=_init_connection, **pool_kwargs)
        return cls(pool)

    async def close(self) -> None:
        await self._pool.close()

    # =========================================================================
    # RAG: HYDRATION
    # =========================================================================

    async def hydrate(
        self,
        query: str,
        *,
        memory_limit: int | None = None,
        include_partial: bool = True,
        include_identity: bool = True,
        include_worldview: bool = True,
        include_emotional_state: bool = True,
        include_goals: bool = False,
        include_drives: bool = True,
        include_subgraph: bool = True,
        subgraph_depth: int = 2,
        subgraph_budget: int = 40,
        subgraph_rel_types: list[str] | None = None,
        session_id: UUID | str | None = None,
        exclude_sensitive: bool = False,
    ) -> HydratedContext:
        """
        Hydrate a query with relevant context for RAG prompt augmentation.

        This uses:
        - `recmem_recall_context(query, limit)` for relevant memories (episodes,
          semantic facts, and recent raw turns — the chat hot path's ranker)
        - `fast_recall(query, limit)` for worldview-typed memories only
        - `find_partial_activations(query)` for tip-of-tongue clusters (optional)
        - `gather_turn_context()` for identity/worldview/emotions/drives/goals (optional subsets)
        """
        import asyncio as _aio

        if memory_limit is None:
            # Budget, not knowledge limit: the default is config-owned (WS6).
            async with self._pool.acquire() as conn:
                memory_limit = int(
                    await conn.fetchval(
                        "SELECT COALESCE(get_config_int('memory.hydrate_memory_limit'), 10)"
                    )
                    or 10
                )

        async def _fetch_memories():
            async with self._pool.acquire() as conn:
                recalled = await self._recall_recmem(
                    conn,
                    query,
                    memory_limit,
                    session_id=session_id,
                    exclude_sensitive=exclude_sensitive,
                )
                relevant_worldview = []
                if include_worldview:
                    relevant_worldview = await self._recall_memories(
                        conn,
                        query,
                        max(10, memory_limit),
                        [MemoryType.WORLDVIEW],
                    )
                return _deduplicate_memories([*relevant_worldview[:3], *recalled])[
                    :memory_limit
                ]

        async def _fetch_partial():
            if not include_partial:
                return []
            async with self._pool.acquire() as conn:
                return await self._find_partial_activations(conn, query)

        async def _fetch_context():
            async with self._pool.acquire() as conn:
                return await conn.fetchval(
                    "SELECT gather_turn_context() || jsonb_build_object('user_model', get_approved_user_model_context(12))"
                )

        memories, partial, ctx_row = await _aio.gather(
            _fetch_memories(), _fetch_partial(), _fetch_context()
        )

        ctx = _coerce_json(ctx_row)

        identity = ctx.get("identity", []) if include_identity else []
        worldview = ctx.get("worldview", []) if include_worldview else []
        emotional_state = (
            ctx.get("emotional_state") if include_emotional_state else None
        )
        goals = ctx.get("goals") if include_goals else None
        urgent_drives = ctx.get("urgent_drives", []) if include_drives else []
        user_model = ctx.get("user_model", [])

        # Seed the dynamic sub-knowledge-graph from the recalled memories. Runs
        # after recall (it depends on the recalled ids), as one small round-trip.
        subgraph = None
        context_paths = None
        if include_subgraph and memories:
            seed_ids = [m.id for m in memories]
            # Default to the semantic reasoning edges (curated); callers can pass
            # an explicit list (incl. structural types) to override.
            rel_types = (
                subgraph_rel_types
                if subgraph_rel_types is not None
                else REASONING_EDGE_TYPES
            )
            try:
                async with self._pool.acquire() as conn:
                    sg_row = await conn.fetchrow(
                        """
                        SELECT
                            build_context_subgraph($1::uuid[], $2, $3::text[], $4) AS subgraph,
                            memory_context_paths($1::uuid[], $2) AS context_paths
                        """,
                        seed_ids,
                        subgraph_depth,
                        rel_types,
                        subgraph_budget,
                    )
                subgraph = _coerce_json(sg_row["subgraph"]) if sg_row else None
                context_paths = (
                    _coerce_json(sg_row["context_paths"]) if sg_row else None
                )
            except Exception:
                # advisory: never fail hydration on graph assembly
                subgraph = None
                context_paths = None

        return HydratedContext(
            memories=memories,
            partial_activations=partial,
            identity=list(identity) if isinstance(identity, list) else [],
            worldview=list(worldview) if isinstance(worldview, list) else [],
            user_model=list(user_model) if isinstance(user_model, list) else [],
            emotional_state=(
                dict(emotional_state) if isinstance(emotional_state, dict) else None
            ),
            goals=dict(goals) if isinstance(goals, dict) else None,
            urgent_drives=(
                list(urgent_drives) if isinstance(urgent_drives, list) else []
            ),
            subgraph=subgraph if isinstance(subgraph, dict) else None,
            context_paths=context_paths if isinstance(context_paths, dict) else None,
        )

    async def hydrate_batch(
        self,
        queries: list[str],
        *,
        max_concurrency: int = 5,
        **kwargs: Any,
    ) -> list[HydratedContext]:
        """
        Hydrate multiple queries concurrently (pool-backed).

        Note: `asyncpg.Connection` cannot run concurrent queries, so batching here
        means concurrent hydrations across pooled connections.
        """
        sem = asyncio.Semaphore(max(1, max_concurrency))

        async def _one(q: str) -> HydratedContext:
            async with sem:
                return await self.hydrate(q, **kwargs)

        return list(await asyncio.gather(*[_one(q) for q in queries]))

    # =========================================================================
    # RECALL
    # =========================================================================

    async def recall(
        self,
        query: str,
        *,
        limit: int = 10,
        memory_types: list[MemoryType] | None = None,
        min_importance: float = 0.0,
        include_partial: bool = True,
    ) -> RecallResult:
        async with self._pool.acquire() as conn:
            memories = await self._recall_memories(
                conn,
                query,
                limit,
                memory_types=memory_types,
                min_importance=min_importance,
            )
            partial = (
                await self._find_partial_activations(conn, query)
                if include_partial
                else []
            )
            return RecallResult(
                memories=memories, partial_activations=partial, query=query
            )

    async def search_history(
        self,
        query: str,
        *,
        limit: int = 20,
        sources: list[str] | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        exclude_session_id: UUID | str | None = None,
    ) -> list[HistorySearchResult]:
        """Run free Postgres FTS across raw turns and consolidated memories.

        An empty query with a created_after/created_before window browses the
        window chronologically (#68) — the SQL treats '' / '*' as browse mode.
        """

        normalized_query = query.strip()
        if (
            not normalized_query.strip("* ")
            and created_after is None
            and created_before is None
        ):
            raise ValueError(
                "history search needs query keywords, or a created_after/"
                "created_before window to browse chronologically"
            )
        normalized_sources = list(
            dict.fromkeys(sources if sources is not None else ["turn", "memory"])
        )
        invalid_sources = sorted(set(normalized_sources) - {"turn", "memory"})
        if invalid_sources:
            raise ValueError(
                "history search sources must be 'turn' and/or 'memory'; invalid: "
                + ", ".join(invalid_sources)
            )
        if not normalized_sources:
            raise ValueError("history search requires at least one source")
        exclude_session = _uuid_text_or_none(exclude_session_id)
        if exclude_session_id is not None and exclude_session is None:
            raise ValueError("exclude_session_id must be a UUID")
        normalized_after = created_after
        normalized_before = created_before
        if normalized_after is not None and normalized_after.tzinfo is None:
            normalized_after = normalized_after.replace(tzinfo=timezone.utc)
        if normalized_before is not None and normalized_before.tzinfo is None:
            normalized_before = normalized_before.replace(tzinfo=timezone.utc)
        if (
            normalized_after is not None
            and normalized_before is not None
            and normalized_after >= normalized_before
        ):
            raise ValueError("created_after must be earlier than created_before")

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT *
                FROM search_cross_session_history(
                    $1::text,
                    $2::int,
                    $3::text[],
                    $4::timestamptz,
                    $5::timestamptz,
                    $6::uuid
                )
                """,
                normalized_query,
                min(max(int(limit), 1), 100),
                normalized_sources,
                normalized_after,
                normalized_before,
                exclude_session,
            )

        results: list[HistorySearchResult] = []
        for row in rows:
            raw_memory_type = row["memory_type"]
            source_attribution = _coerce_json(row["source_attribution"])
            metadata = _coerce_json(row["metadata"])
            results.append(
                HistorySearchResult(
                    source_kind=str(row["source_kind"]),
                    item_id=row["item_id"],
                    session_id=row["session_id"],
                    content=str(row["content"]),
                    user_text=row["user_text"],
                    assistant_text=row["assistant_text"],
                    memory_type=(
                        MemoryType(str(raw_memory_type))
                        if raw_memory_type is not None
                        else None
                    ),
                    occurred_at=row["occurred_at"],
                    rank=float(row["rank"]),
                    source_unit_ids=list(row["source_unit_ids"] or []),
                    source_attribution=(
                        dict(source_attribution)
                        if isinstance(source_attribution, dict)
                        else {}
                    ),
                    metadata=dict(metadata) if isinstance(metadata, dict) else {},
                )
            )
        return results

    async def recall_by_id(self, memory_id: UUID) -> Memory | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    id,
                    type,
                    content,
                    importance,
                    trust_level,
                    source_attribution,
                    created_at,
                    emotional_valence
                FROM get_memory_by_id($1::uuid)
                """,
                memory_id,
            )
            if not row:
                return None
            return Memory(
                id=row["id"],
                type=MemoryType(row["type"]),
                content=row["content"],
                importance=float(row["importance"]),
                trust_level=(
                    float(row["trust_level"])
                    if row["trust_level"] is not None
                    else None
                ),
                source_attribution=(
                    _coerce_json(row["source_attribution"])
                    if row["source_attribution"] is not None
                    else None
                ),
                created_at=row["created_at"],
                emotional_valence=row["emotional_valence"],
            )

    async def recall_recent(
        self,
        *,
        limit: int = 10,
        memory_type: MemoryType | None = None,
    ) -> list[Memory]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    memory_id as id,
                    memory_type as type,
                    content,
                    importance,
                    trust_level,
                    source_attribution,
                    created_at,
                    emotional_valence
                FROM list_recent_memories($1::int, $2::memory_type[], $3::bool)
                """,
                limit,
                [memory_type.value] if memory_type is not None else None,
                False,
            )
            return [self._row_to_memory(row) for row in rows]

    async def list_recent_episodes(self, *, limit: int = 5) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    id,
                    started_at,
                    ended_at,
                    episode_type,
                    summary,
                    memory_count
                FROM list_recent_episodes($1::int)
                """,
                limit,
            )
            return [dict(r) for r in rows]

    async def recall_episode(self, episode_id: UUID) -> list[Memory]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    memory_id as id,
                    memory_type as type,
                    content,
                    importance,
                    trust_level,
                    source_attribution,
                    created_at,
                    emotional_valence
                FROM get_episode_memories($1::uuid)
                """,
                episode_id,
            )
            return [self._row_to_memory(row) for row in rows]

    # =========================================================================
    # REMEMBER
    # =========================================================================

    async def remember(
        self,
        content: str,
        *,
        type: MemoryType = MemoryType.EPISODIC,
        importance: float = 0.5,
        emotional_valence: float = 0.0,
        context: dict[str, Any] | None = None,
        concepts: list[str] | None = None,
        source_attribution: dict[str, Any] | None = None,
        source_references: Any | None = None,
        trust_level: float | None = None,
    ) -> UUID:
        async with self._pool.acquire() as conn:
            memory_id = await self._create_memory(
                conn,
                content,
                type,
                importance,
                emotional_valence,
                context,
                source_attribution=source_attribution,
                source_references=source_references,
                trust_level=trust_level,
            )

            if concepts:
                await conn.executemany(
                    "SELECT link_memory_to_concept($1::uuid, $2::text, 1.0)",
                    [(memory_id, c) for c in concepts],
                )

            return memory_id

    async def remember_turn_raw(
        self,
        user_text: str,
        assistant_text: str,
        *,
        session_id: UUID | str | None = None,
        source_identity: str | None = None,
        turn_at: datetime | None = None,
        importance: float = 0.3,
        source_attribution: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        async with self._pool.acquire() as conn:
            raw = await conn.fetchval(
                """
                SELECT recmem_ingest_turn(
                    $1::text,
                    $2::text,
                    $3::uuid,
                    $4::text,
                    COALESCE($5::timestamptz, CURRENT_TIMESTAMP),
                    $6::float,
                    $7::jsonb,
                    $8::jsonb
                )
                """,
                user_text,
                assistant_text,
                _uuid_text_or_none(session_id),
                source_identity,
                turn_at,
                float(importance),
                _to_jsonb_arg(source_attribution),
                _to_jsonb_arg(metadata or {}),
            )
            result = _coerce_json(raw) if raw is not None else {}
            return dict(result) if isinstance(result, dict) else {}

    async def record_chat_turn_memory(
        self,
        user_text: str,
        assistant_text: str,
        *,
        session_id: UUID | str | None = None,
        source_identity: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        async with self._pool.acquire() as conn:
            raw = await conn.fetchval(
                """
                SELECT record_chat_turn_memory(
                    $1::text,
                    $2::text,
                    $3::text,
                    $4::text,
                    $5::jsonb
                )
                """,
                user_text,
                assistant_text,
                str(session_id) if session_id is not None else None,
                source_identity,
                _to_jsonb_arg(context or {}),
            )
            result = _coerce_json(raw) if raw is not None else {}
            return dict(result) if isinstance(result, dict) else {}

    async def hydrate_chat_session(
        self,
        session_id: UUID | str,
        *,
        limit: int | None = None,
        include_system: bool = False,
    ) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT hydrate_chat_session($1::uuid, $2::int, $3::bool)",
                _uuid_text_or_none(session_id),
                limit,
                include_system,
            )
            result = _coerce_json(raw) if raw is not None else {}
            messages = result.get("messages", []) if isinstance(result, dict) else []
            return list(messages) if isinstance(messages, list) else []

    async def record_chat_session_turn(
        self,
        user_text: str,
        assistant_text: str,
        *,
        session_id: UUID | str,
        surface: str = "chat",
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        async with self._pool.acquire() as conn:
            raw = await conn.fetchval(
                """
                SELECT record_chat_session_turn(
                    $1::uuid,
                    $2::text,
                    $3::text,
                    $4::text,
                    $5::jsonb
                )
                """,
                _uuid_text_or_none(session_id),
                user_text,
                assistant_text,
                surface,
                _to_jsonb_arg(context or {}),
            )
            result = _coerce_json(raw) if raw is not None else {}
            return dict(result) if isinstance(result, dict) else {}

    async def clear_chat_session_context(
        self,
        session_id: UUID | str,
        *,
        reason: str | None = None,
    ) -> dict[str, Any]:
        async with self._pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT clear_chat_session_context($1::uuid, $2::text)",
                _uuid_text_or_none(session_id),
                reason,
            )
            result = _coerce_json(raw) if raw is not None else {}
            return dict(result) if isinstance(result, dict) else {}

    async def hydrate_recmem(
        self,
        query: str,
        *,
        sub_limit: int | None = None,
        epi_limit: int | None = None,
        sem_limit: int | None = None,
        session_id: UUID | str | None = None,
        exclude_sensitive: bool = False,
    ) -> list[Memory]:
        async with self._pool.acquire() as conn:
            return await self._recall_recmem(
                conn,
                query,
                max(sub_limit or 10, (epi_limit or 5) + (sem_limit or 10)),
                sub_limit=sub_limit,
                epi_limit=epi_limit,
                sem_limit=sem_limit,
                session_id=session_id,
            )

    async def link_to_source_unit(
        self,
        memory_id: UUID | str,
        unit_id: UUID | str,
        role: str = "direct_promotion",
    ) -> bool:
        async with self._pool.acquire() as conn:
            return bool(
                await conn.fetchval(
                    "SELECT link_memory_to_source_unit($1::uuid, $2::uuid, $3::text)",
                    str(memory_id),
                    str(unit_id),
                    role,
                )
            )

    async def redact_unit(
        self,
        unit_id: UUID | str,
        *,
        reason: str | None = None,
        cascade: bool = True,
    ) -> dict[str, Any]:
        async with self._pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT recmem_redact_unit($1::uuid, $2::text, $3::boolean)",
                str(unit_id),
                reason,
                cascade,
            )
            result = _coerce_json(raw) if raw is not None else {}
            return dict(result) if isinstance(result, dict) else {}

    async def add_source(self, memory_id: UUID, source: dict[str, Any]) -> None:
        """Attach an additional source reference to a semantic memory and recompute trust."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "SELECT add_semantic_source_reference($1::uuid, $2::jsonb)",
                memory_id,
                _to_jsonb_arg(source),
            )

    async def get_truth_profile(self, memory_id: UUID) -> dict[str, Any]:
        """Return DB-computed provenance/trust details for a memory."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT get_memory_truth_profile($1::uuid) AS profile", memory_id
            )
            if not row or row["profile"] is None:
                return {}
            return dict(_coerce_json(row["profile"]))

    async def remember_batch(self, memories: Iterable[MemoryInput]) -> list[UUID]:
        async with self._pool.acquire() as conn:
            items: list[dict[str, Any]] = []
            mem_list = list(memories)
            for m in mem_list:
                item: dict[str, Any] = {
                    "type": m.type.value,
                    "content": m.content,
                    "importance": m.importance,
                }
                if m.source_attribution is not None:
                    item["source_attribution"] = m.source_attribution
                if m.trust_level is not None:
                    item["trust_level"] = m.trust_level
                if m.type == MemoryType.EPISODIC:
                    item["context"] = m.context
                    item["emotional_valence"] = m.emotional_valence
                elif m.type == MemoryType.SEMANTIC:
                    item["source_references"] = (
                        m.source_references
                        if m.source_references is not None
                        else m.context
                    )
                elif m.type == MemoryType.PROCEDURAL:
                    item["steps"] = (
                        m.context if m.context is not None else {"steps": []}
                    )
                elif m.type == MemoryType.STRATEGIC:
                    item["supporting_evidence"] = m.context
                items.append(item)

            import json

            created = await conn.fetchval(
                "SELECT batch_create_memories($1::jsonb)", json.dumps(items)
            )
            ids = list(created or [])

            # Link concepts in batch
            concept_pairs = []
            for mid, m in zip(ids, mem_list):
                if m.concepts:
                    concept_pairs.extend((mid, c) for c in m.concepts)
            if concept_pairs:
                await conn.executemany(
                    "SELECT link_memory_to_concept($1::uuid, $2::text, 1.0)",
                    concept_pairs,
                )

            return ids

    async def remember_batch_raw(
        self,
        contents: list[str],
        embeddings: list[list[float]],
        *,
        type: MemoryType = MemoryType.EPISODIC,
        importance: float = 0.5,
    ) -> list[UUID]:
        """
        Insert memories with pre-computed embeddings (bypasses get_embedding()).

        Notes:
        - Graph nodes are created to keep AGE state consistent.
        - Embedding dimension must match the DB typmod.
        """
        if len(contents) != len(embeddings):
            raise ValueError("contents and embeddings must have same length")

        async with self._pool.acquire() as conn:
            expected_dim = int(await conn.fetchval("SELECT embedding_dimension()"))
            for embedding in embeddings:
                if len(embedding) != expected_dim:
                    raise ValueError(
                        f"embedding dimension mismatch: expected {expected_dim}, got {len(embedding)}"
                    )

            created = await conn.fetchval(
                """
                SELECT batch_create_memories_with_embeddings(
                    $1::memory_type,
                    $2::text[],
                    $3::jsonb,
                    $4::float
                )
                """,
                type.value,
                contents,
                _to_jsonb_arg(embeddings),
                float(importance),
            )
            return list(created or [])

    async def touch_memories(self, memory_ids: Iterable[UUID]) -> int:
        """Increment access_count/last_accessed for the given memory ids."""
        ids = list(memory_ids)
        if not ids:
            return 0
        async with self._pool.acquire() as conn:
            updated = await conn.fetchval("SELECT touch_memories($1::uuid[])", ids)
            return int(updated or 0)

    # =========================================================================
    # GRAPH / RELATIONSHIPS
    # =========================================================================

    async def connect_memories(
        self,
        from_id: UUID,
        to_id: UUID,
        relationship: RelationshipType,
        *,
        confidence: float = 0.8,
        context: str | None = None,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                SELECT discover_relationship($1::uuid, $2::uuid, $3::graph_edge_type, $4::float, 'api', NULL, $5::text)
                """,
                from_id,
                to_id,
                relationship.value,
                confidence,
                context,
            )

    async def connect_batch(self, relationships: Iterable[RelationshipInput]) -> None:
        rel_list = list(relationships)
        if not rel_list:
            return
        async with self._pool.acquire() as conn:
            await conn.executemany(
                "SELECT discover_relationship($1::uuid, $2::uuid, $3::graph_edge_type, $4::float, 'api', NULL, $5::text)",
                [
                    (
                        r.from_id,
                        r.to_id,
                        r.relationship_type.value,
                        r.confidence,
                        r.context,
                    )
                    for r in rel_list
                ],
            )

    async def find_causes(
        self, memory_id: UUID, *, depth: int = 3
    ) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM find_causal_chain($1::uuid, $2::int)", memory_id, depth
            )
            return [dict(row) for row in rows]

    async def find_contradictions(
        self, memory_id: UUID | None = None
    ) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM find_contradictions($1::uuid)", memory_id
            )
            return [dict(row) for row in rows]

    async def find_supporting_evidence(
        self, worldview_id: UUID
    ) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM find_supporting_evidence($1::uuid)", worldview_id
            )
            return [dict(row) for row in rows]

    # =========================================================================
    # CONCEPTS
    # =========================================================================

    async def link_concept(
        self, memory_id: UUID, concept: str, *, strength: float = 1.0
    ) -> UUID:
        async with self._pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT link_memory_to_concept($1::uuid, $2::text, $3::float)",
                memory_id,
                concept,
                strength,
            )

    async def find_by_concept(self, concept: str, *, limit: int = 10) -> list[Memory]:
        """Find memories linked to a concept via graph traversal.
        Phase 2 (ReduceScopeCreep): Now uses graph instead of relational tables.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT memory_id as id, memory_type as type, memory_content as content,
                       memory_importance as importance, memory_created_at as created_at,
                       emotional_valence
                FROM find_memories_by_concept($1::text, $2::int)
                """,
                concept,
                limit,
            )
            return [self._row_to_memory(row) for row in rows]

    # =========================================================================
    # WORKING MEMORY
    # =========================================================================

    async def hold(self, content: str, *, ttl_seconds: int = 3600) -> UUID:
        async with self._pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT add_to_working_memory($1::text, $2::int * interval '1 second')",
                content,
                ttl_seconds,
            )

    async def search_working(
        self, query: str, *, limit: int = 5
    ) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM search_working_memory($1::text, $2::int)", query, limit
            )
            return [dict(row) for row in rows]

    # =========================================================================
    # STATE / INTROSPECTION
    # =========================================================================

    async def get_emotional_state(self) -> dict[str, Any] | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM current_emotional_state")
            return dict(row) if row else None

    async def sense_memory_availability(self, query: str) -> dict[str, Any]:
        async with self._pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT sense_memory_availability($1::text)", query
            )
            return _coerce_json(raw) if raw is not None else {}

    async def request_background_search(self, query: str) -> UUID | None:
        async with self._pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT request_background_search($1::text)", query
            )

    async def get_spontaneous_memories(self, *, limit: int = 3) -> list[Memory]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    id,
                    type,
                    content,
                    importance,
                    trust_level,
                    source_attribution,
                    created_at,
                    (metadata->>'emotional_valence')::float as emotional_valence
                FROM get_spontaneous_memories($1::int)
                """,
                limit,
            )
            return [self._row_to_memory(row) for row in rows]

    async def get_drives(self) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM drive_status")
            return [dict(row) for row in rows]

    async def get_health(self) -> dict[str, Any]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM cognitive_health")
            return dict(row) if row else {}

    async def get_identity(self) -> list[dict[str, Any]]:
        """Get identity aspects from graph (Phase 5: uses graph instead of identity_aspects table)."""
        async with self._pool.acquire() as conn:
            # Phase 5: Identity aspects are now graph edges from SelfNode
            rows = await conn.fetch("""
                SELECT * FROM get_identity_context()
                """)
            result = rows[0][0] if rows and rows[0] else []
            return result if isinstance(result, list) else []

    async def get_worldview(self) -> list[dict[str, Any]]:
        """Get worldview beliefs from memories (Phase 5: uses worldview memories instead of worldview_primitives table)."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM get_worldview_snapshot($1::int, $2::float)", 5, 0.5
            )
            return [dict(row) for row in rows]

    async def get_goals(
        self, *, priority: GoalPriority | None = None
    ) -> list[dict[str, Any]]:
        """Get goals by priority.

        Phase 6 (ReduceScopeCreep): Goals are now memories with type='goal'.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM get_goals_by_priority($1::goal_priority)",
                priority.value if priority is not None else None,
            )
            return [dict(row) for row in rows]

    async def create_goal(
        self,
        title: str,
        *,
        description: str | None = None,
        source: GoalSource | str = GoalSource.USER_REQUEST,
        origin: GoalSource | str | None = None,
        priority: GoalPriority | str = GoalPriority.QUEUED,
        parent_id: UUID | None = None,
        due_at: datetime | None = None,
    ) -> UUID:
        async with self._pool.acquire() as conn:
            return await conn.fetchval(
                """
                SELECT create_goal(
                    $1,
                    $2,
                    $3::goal_source,
                    $4::goal_priority,
                    $5::uuid,
                    $6::timestamptz,
                    $7::goal_source
                )
                """,
                title,
                description,
                (source.value if isinstance(source, GoalSource) else str(source)),
                (
                    priority.value
                    if isinstance(priority, GoalPriority)
                    else str(priority)
                ),
                parent_id,
                due_at,
                (
                    origin.value
                    if isinstance(origin, GoalSource)
                    else (
                        str(origin)
                        if origin is not None
                        else (
                            source.value
                            if isinstance(source, GoalSource)
                            else str(source)
                        )
                    )
                ),
            )

    async def create_scheduled_task(
        self,
        name: str,
        *,
        schedule_kind: str,
        schedule: dict[str, Any],
        action_kind: str,
        action_payload: dict[str, Any] | None = None,
        timezone: str | None = None,
        description: str | None = None,
        status: str | None = None,
        max_runs: int | None = None,
        created_by: str | None = None,
    ) -> UUID:
        async with self._pool.acquire() as conn:
            return await conn.fetchval(
                """
                SELECT create_scheduled_task(
                    $1,
                    $2,
                    $3::jsonb,
                    $4,
                    $5::jsonb,
                    $6,
                    $7,
                    $8,
                    $9,
                    $10
                )
                """,
                name,
                schedule_kind,
                _to_jsonb_arg(schedule),
                action_kind,
                _to_jsonb_arg(action_payload or {}),
                timezone,
                description,
                status,
                max_runs,
                created_by,
            )

    async def list_scheduled_tasks(
        self,
        *,
        status: str | None = None,
        due_before: datetime | str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM list_scheduled_tasks($1, $2::timestamptz, $3)",
                status,
                due_before,
                int(limit),
            )
            return [dict(row) for row in rows]

    async def update_scheduled_task(
        self,
        task_id: UUID,
        *,
        name: str | None = None,
        description: str | None = None,
        schedule_kind: str | None = None,
        schedule: dict[str, Any] | None = None,
        timezone: str | None = None,
        action_kind: str | None = None,
        action_payload: dict[str, Any] | None = None,
        status: str | None = None,
        max_runs: int | None = None,
    ) -> dict[str, Any]:
        async with self._pool.acquire() as conn:
            raw = await conn.fetchval(
                """
                SELECT update_scheduled_task(
                    $1::uuid,
                    $2,
                    $3,
                    $4,
                    $5::jsonb,
                    $6,
                    $7,
                    $8::jsonb,
                    $9,
                    $10
                )
                """,
                task_id,
                name,
                description,
                schedule_kind,
                _to_jsonb_arg(schedule),
                timezone,
                action_kind,
                _to_jsonb_arg(action_payload),
                status,
                max_runs,
            )
            return _coerce_json(raw) if raw is not None else {}

    async def delete_scheduled_task(
        self,
        task_id: UUID,
        *,
        hard_delete: bool = False,
        reason: str | None = None,
    ) -> bool:
        async with self._pool.acquire() as conn:
            return bool(
                await conn.fetchval(
                    "SELECT delete_scheduled_task($1::uuid, $2::boolean, $3)",
                    task_id,
                    hard_delete,
                    reason,
                )
            )

    async def queue_user_message(
        self,
        message: str,
        *,
        intent: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        async with self._pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT build_user_message($1, $2, $3::jsonb)",
                message,
                intent,
                _to_jsonb_arg(context or {}),
            )
            return _coerce_json(raw)

    async def get_ingestion_receipts(
        self, source_file: str, content_hashes: list[str]
    ) -> dict[str, UUID]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    (m.source_attribution->>'content_hash')::text AS content_hash,
                    m.id AS memory_id
                FROM memories m
                WHERE m.source_attribution->>'ref' = $1
                  AND m.source_attribution->>'content_hash' = ANY($2::text[])
                """,
                source_file,
                content_hashes,
            )
            out: dict[str, UUID] = {}
            for r in rows:
                try:
                    out[str(r["content_hash"])] = UUID(str(r["memory_id"]))
                except Exception:
                    continue
            return out

    async def record_ingestion_receipts(self, items: list[dict[str, Any]]) -> int:
        # No-op: ingestion receipts are implicit in memories.source_attribution.
        return int(len(items or []))

    # =========================================================================
    # INTERNALS
    # =========================================================================

    async def _create_memory(
        self,
        conn: asyncpg.Connection,
        content: str,
        type: MemoryType,
        importance: float,
        emotional_valence: float,
        context: dict[str, Any] | None,
        *,
        source_attribution: dict[str, Any] | None = None,
        source_references: Any | None = None,
        trust_level: float | None = None,
    ) -> UUID:
        if type == MemoryType.EPISODIC:
            return await conn.fetchval(
                "SELECT create_episodic_memory($1::text, NULL, $2::jsonb, NULL, $3::float, CURRENT_TIMESTAMP, $4::float, $5::jsonb, $6::float)",
                content,
                _to_jsonb_arg(context),
                emotional_valence,
                importance,
                _to_jsonb_arg(source_attribution),
                trust_level,
            )
        if type == MemoryType.SEMANTIC:
            sources = source_references if source_references is not None else context
            return await conn.fetchval(
                "SELECT create_semantic_memory($1::text, 0.8::float, NULL, NULL, $2::jsonb, $3::float, $4::jsonb, $5::float)",
                content,
                _to_jsonb_arg(sources),
                importance,
                _to_jsonb_arg(source_attribution),
                trust_level,
            )
        if type == MemoryType.PROCEDURAL:
            steps = context if context is not None else {}
            return await conn.fetchval(
                "SELECT create_procedural_memory($1::text, $2::jsonb, NULL, $3::float, $4::jsonb, $5::float)",
                content,
                _to_jsonb_arg(steps),
                importance,
                _to_jsonb_arg(source_attribution),
                trust_level,
            )
        if type == MemoryType.STRATEGIC:
            return await conn.fetchval(
                "SELECT create_strategic_memory($1::text, $2::text, 0.8::float, $3::jsonb, NULL, $4::float, $5::jsonb, $6::float)",
                content,
                content,
                _to_jsonb_arg(context),
                importance,
                _to_jsonb_arg(source_attribution),
                trust_level,
            )
        raise ValueError(f"Unknown memory type: {type}")

    async def _recall_memories(
        self,
        conn: asyncpg.Connection,
        query: str,
        limit: int,
        memory_types: list[MemoryType] | None = None,
        min_importance: float = 0.0,
    ) -> list[Memory]:
        rows = await conn.fetch(
            """
            SELECT
                memory_id,
                content,
                memory_type,
                score,
                source,
                importance,
                trust_level,
                source_attribution,
                created_at,
                emotional_valence
            FROM recall_memories_filtered($1::text, $2::int, $3::memory_type[], $4::float)
            """,
            query,
            limit,
            [mt.value for mt in memory_types] if memory_types else None,
            min_importance,
        )

        memories: list[Memory] = []
        for row in rows:
            mt = MemoryType(row["memory_type"])
            if memory_types is not None and mt not in set(memory_types):
                continue
            memories.append(
                Memory(
                    id=row["memory_id"],
                    type=mt,
                    content=row["content"],
                    importance=float(row["importance"]),
                    similarity=float(row["score"]),
                    source=row["source"],
                    trust_level=(
                        float(row["trust_level"])
                        if row["trust_level"] is not None
                        else None
                    ),
                    source_attribution=(
                        _coerce_json(row["source_attribution"])
                        if row["source_attribution"] is not None
                        else None
                    ),
                    created_at=row["created_at"],
                    emotional_valence=row["emotional_valence"],
                )
            )
        return memories

    async def _recall_recmem(
        self,
        conn: asyncpg.Connection,
        query: str,
        limit: int,
        *,
        sub_limit: int | None = None,
        epi_limit: int | None = None,
        sem_limit: int | None = None,
        session_id: UUID | str | None = None,
        exclude_sensitive: bool = False,
    ) -> list[Memory]:
        # Per-tier budgets are config-owned (#76): memory.recmem_sub/epi/sem_limit
        # seed the ceilings; the caller's overall limit still scales below them.
        cfg = await conn.fetchrow("""
            SELECT COALESCE(get_config_int('memory.recmem_sub_limit'), 10) AS sub,
                   COALESCE(get_config_int('memory.recmem_epi_limit'), 5) AS epi,
                   COALESCE(get_config_int('memory.recmem_sem_limit'), 10) AS sem,
                   COALESCE(get_config_int('memory.recmem_know_limit'), 5) AS know
            """)
        rows = await conn.fetch(
            """
            SELECT *
            FROM recmem_recall_context(
                $1::text,
                $2::int,
                $3::int,
                $4::int,
                $5::uuid,
                $6::boolean,
                $7::int
            )
            """,
            query,
            int(sub_limit if sub_limit is not None else min(max(limit, 1), cfg["sub"])),
            int(epi_limit if epi_limit is not None else max(1, min(limit, cfg["epi"]))),
            int(
                sem_limit
                if sem_limit is not None
                else max(1, min(limit * 2, cfg["sem"]))
            ),
            _uuid_text_or_none(session_id),
            exclude_sensitive,
            int(cfg["know"]),
        )

        derived_sources: set[UUID] = set()
        for row in rows:
            if row["tier"] != "subconscious":
                derived_sources.update(row["source_unit_ids"] or [])

        memories: list[Memory] = []
        for row in rows:
            if row["tier"] == "subconscious" and row["item_id"] in derived_sources:
                continue
            raw_type = row["memory_type"] or MemoryType.EPISODIC.value
            memories.append(
                Memory(
                    id=row["item_id"],
                    type=MemoryType(raw_type),
                    content=row["content"],
                    importance=0.3,
                    similarity=(
                        float(row["score"]) if row["score"] is not None else None
                    ),
                    source="recmem",
                    trust_level=(
                        float(row["trust_level"])
                        if row["trust_level"] is not None
                        else None
                    ),
                    source_attribution=(
                        _coerce_json(row["source_attribution"])
                        if row["source_attribution"] is not None
                        else None
                    ),
                    created_at=row["created_at"],
                    tier=row["tier"],
                    source_unit_ids=list(row["source_unit_ids"] or []),
                    strength=(
                        float(row["strength"]) if row["strength"] is not None else None
                    ),
                    fidelity=(
                        float(row["fidelity"]) if row["fidelity"] is not None else None
                    ),
                    emotional_intensity=(
                        float(row["emotional_intensity"])
                        if row["emotional_intensity"] is not None
                        else None
                    ),
                    confidence=(
                        float(row["confidence"])
                        if row["confidence"] is not None
                        else None
                    ),
                )
            )

        memories = _deduplicate_memories(memories)

        # Reinforce-on-recall: recalling a memory strengthens it (resets its decay
        # clock -- the up-ladder). Only episodic/semantic tiers are `memories`
        # rows; subconscious item_ids are raw units. The chat/hydrate path did not
        # reinforce before this. Advisory -- never fail recall on it.
        recalled_ids = [
            memory.id for memory in memories if memory.tier != "subconscious"
        ]
        if recalled_ids:
            try:
                await conn.execute("SELECT touch_memories($1::uuid[])", recalled_ids)
            except Exception as exc:
                logger.warning("Failed to mark recalled memories as accessed: %s", exc)
        raw_unit_ids = [
            memory.id for memory in memories if memory.tier == "subconscious"
        ]
        if raw_unit_ids:
            try:
                await conn.execute(
                    "SELECT touch_subconscious_units($1::uuid[])", raw_unit_ids
                )
            except Exception as exc:
                logger.warning(
                    "Failed to mark recalled RecMem units as accessed: %s", exc
                )
        return memories

    async def _find_partial_activations(
        self, conn: asyncpg.Connection, query: str
    ) -> list[PartialActivation]:
        rows = await conn.fetch(
            "SELECT * FROM find_partial_activations($1::text)", query
        )
        out: list[PartialActivation] = []
        for row in rows:
            out.append(
                PartialActivation(
                    cluster_id=row["cluster_id"],
                    cluster_name=row["cluster_name"],
                    keywords=list(row["keywords"] or []),
                    emotional_signature=(
                        _coerce_json(row["emotional_signature"])
                        if row["emotional_signature"] is not None
                        else None
                    ),
                    cluster_similarity=float(row["cluster_similarity"]),
                    best_memory_similarity=float(row["best_memory_similarity"]),
                )
            )
        return out

    async def explore_clusters(
        self, query: str, limit: int = 3, sample_size: int = 3
    ) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM explore_clusters_with_samples($1::text, $2::int, $3::int)",
                query,
                limit,
                sample_size,
            )
            clusters_map: dict[str, dict[str, Any]] = {}
            for row in rows:
                cid = str(row["cluster_id"])
                if cid not in clusters_map:
                    clusters_map[cid] = {
                        "id": row["cluster_id"],
                        "name": row["cluster_name"],
                        "cluster_type": row["cluster_type"],
                        "similarity": row["cluster_similarity"],
                        "sample_memories": [],
                    }
                if row["memory_id"] is not None:
                    clusters_map[cid]["sample_memories"].append(
                        {
                            "memory_id": row["memory_id"],
                            "content": row["content"],
                            "memory_type": row["memory_type"],
                            "membership_strength": row["membership_strength"],
                        }
                    )
        return list(clusters_map.values())

    def _row_to_memory(self, row: asyncpg.Record) -> Memory:
        return Memory(
            id=row["id"],
            type=MemoryType(row["type"]),
            content=row["content"],
            importance=float(row["importance"]),
            trust_level=(
                float(row["trust_level"])
                if "trust_level" in row and row["trust_level"] is not None
                else None
            ),
            source_attribution=(
                _coerce_json(row["source_attribution"])
                if "source_attribution" in row and row["source_attribution"] is not None
                else None
            ),
            created_at=row["created_at"] if "created_at" in row else None,
            emotional_valence=(
                row["emotional_valence"] if "emotional_valence" in row else None
            ),
            tier=row["tier"] if "tier" in row else None,
            source_unit_ids=(
                list(row["source_unit_ids"] or []) if "source_unit_ids" in row else None
            ),
            valid_until=row["valid_until"] if "valid_until" in row else None,
        )


class CognitiveMemorySync:
    """Synchronous wrapper around CognitiveMemory for non-async call sites."""

    def __init__(self, async_client: CognitiveMemory, loop: asyncio.AbstractEventLoop):
        self._async = async_client
        self._loop = loop

    @classmethod
    def connect(cls, dsn: str, **kwargs: Any) -> "CognitiveMemorySync":
        loop = asyncio.new_event_loop()
        try:
            client = loop.run_until_complete(CognitiveMemory.create(dsn, **kwargs))
        except Exception:
            loop.close()
            raise
        return cls(client, loop)

    def close(self) -> None:
        self._loop.run_until_complete(self._async.close())
        self._loop.close()

    def hydrate(self, query: str, **kwargs: Any) -> HydratedContext:
        return self._loop.run_until_complete(self._async.hydrate(query, **kwargs))

    def recall(self, query: str, **kwargs: Any) -> RecallResult:
        return self._loop.run_until_complete(self._async.recall(query, **kwargs))

    def search_history(self, query: str, **kwargs: Any) -> list[HistorySearchResult]:
        return self._loop.run_until_complete(
            self._async.search_history(query, **kwargs)
        )

    def recall_recent(
        self, *, limit: int = 10, memory_type: MemoryType | None = None
    ) -> list[Memory]:
        return self._loop.run_until_complete(
            self._async.recall_recent(limit=limit, memory_type=memory_type)
        )

    def list_recent_episodes(self, *, limit: int = 5) -> list[dict[str, Any]]:
        return self._loop.run_until_complete(
            self._async.list_recent_episodes(limit=limit)
        )

    def recall_episode(self, episode_id: UUID) -> list[Memory]:
        return self._loop.run_until_complete(self._async.recall_episode(episode_id))

    def remember(self, content: str, **kwargs: Any) -> UUID:
        return self._loop.run_until_complete(self._async.remember(content, **kwargs))

    def remember_turn_raw(
        self, user_text: str, assistant_text: str, **kwargs: Any
    ) -> dict[str, Any]:
        return self._loop.run_until_complete(
            self._async.remember_turn_raw(user_text, assistant_text, **kwargs)
        )

    def hydrate_chat_session(
        self, session_id: UUID | str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        return self._loop.run_until_complete(
            self._async.hydrate_chat_session(session_id, **kwargs)
        )

    def record_chat_session_turn(
        self, user_text: str, assistant_text: str, **kwargs: Any
    ) -> dict[str, Any]:
        return self._loop.run_until_complete(
            self._async.record_chat_session_turn(user_text, assistant_text, **kwargs)
        )

    def clear_chat_session_context(
        self, session_id: UUID | str, **kwargs: Any
    ) -> dict[str, Any]:
        return self._loop.run_until_complete(
            self._async.clear_chat_session_context(session_id, **kwargs)
        )

    def hydrate_recmem(self, query: str, **kwargs: Any) -> list[Memory]:
        return self._loop.run_until_complete(
            self._async.hydrate_recmem(query, **kwargs)
        )

    def link_to_source_unit(
        self, memory_id: UUID | str, unit_id: UUID | str, role: str = "direct_promotion"
    ) -> bool:
        return self._loop.run_until_complete(
            self._async.link_to_source_unit(memory_id, unit_id, role)
        )

    def redact_unit(self, unit_id: UUID | str, **kwargs: Any) -> dict[str, Any]:
        return self._loop.run_until_complete(self._async.redact_unit(unit_id, **kwargs))

    def remember_batch(self, memories: Iterable[MemoryInput]) -> list[UUID]:
        return self._loop.run_until_complete(self._async.remember_batch(memories))

    def remember_batch_raw(
        self, contents: list[str], embeddings: list[list[float]], **kwargs: Any
    ) -> list[UUID]:
        return self._loop.run_until_complete(
            self._async.remember_batch_raw(contents, embeddings, **kwargs)
        )

    def connect_memories(
        self, from_id: UUID, to_id: UUID, relationship: RelationshipType, **kwargs: Any
    ) -> None:
        return self._loop.run_until_complete(
            self._async.connect_memories(from_id, to_id, relationship, **kwargs)
        )

    def link_concept(
        self, memory_id: UUID, concept: str, *, strength: float = 1.0
    ) -> UUID:
        return self._loop.run_until_complete(
            self._async.link_concept(memory_id, concept, strength=strength)
        )

    def connect_batch(self, relationships: "Iterable[RelationshipInput]") -> None:
        return self._loop.run_until_complete(self._async.connect_batch(relationships))

    def touch_memories(self, memory_ids: Iterable[UUID]) -> int:
        return self._loop.run_until_complete(self._async.touch_memories(memory_ids))

    def create_goal(
        self,
        title: str,
        *,
        description: str | None = None,
        source: GoalSource | str = GoalSource.USER_REQUEST,
        origin: GoalSource | str | None = None,
        priority: GoalPriority | str = GoalPriority.QUEUED,
        parent_id: UUID | None = None,
        due_at: datetime | None = None,
    ) -> UUID:
        return self._loop.run_until_complete(
            self._async.create_goal(
                title,
                description=description,
                source=source,
                origin=origin,
                priority=priority,
                parent_id=parent_id,
                due_at=due_at,
            )
        )

    def create_scheduled_task(
        self,
        name: str,
        *,
        schedule_kind: str,
        schedule: dict[str, Any],
        action_kind: str,
        action_payload: dict[str, Any] | None = None,
        timezone: str | None = None,
        description: str | None = None,
        status: str | None = None,
        max_runs: int | None = None,
        created_by: str | None = None,
    ) -> UUID:
        return self._loop.run_until_complete(
            self._async.create_scheduled_task(
                name,
                schedule_kind=schedule_kind,
                schedule=schedule,
                action_kind=action_kind,
                action_payload=action_payload,
                timezone=timezone,
                description=description,
                status=status,
                max_runs=max_runs,
                created_by=created_by,
            )
        )

    def list_scheduled_tasks(
        self,
        *,
        status: str | None = None,
        due_before: datetime | str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return self._loop.run_until_complete(
            self._async.list_scheduled_tasks(
                status=status, due_before=due_before, limit=limit
            )
        )

    def update_scheduled_task(
        self,
        task_id: UUID,
        *,
        name: str | None = None,
        description: str | None = None,
        schedule_kind: str | None = None,
        schedule: dict[str, Any] | None = None,
        timezone: str | None = None,
        action_kind: str | None = None,
        action_payload: dict[str, Any] | None = None,
        status: str | None = None,
        max_runs: int | None = None,
    ) -> dict[str, Any]:
        return self._loop.run_until_complete(
            self._async.update_scheduled_task(
                task_id,
                name=name,
                description=description,
                schedule_kind=schedule_kind,
                schedule=schedule,
                timezone=timezone,
                action_kind=action_kind,
                action_payload=action_payload,
                status=status,
                max_runs=max_runs,
            )
        )

    def delete_scheduled_task(
        self,
        task_id: UUID,
        *,
        hard_delete: bool = False,
        reason: str | None = None,
    ) -> bool:
        return self._loop.run_until_complete(
            self._async.delete_scheduled_task(
                task_id,
                hard_delete=hard_delete,
                reason=reason,
            )
        )

    def queue_user_message(
        self,
        message: str,
        *,
        intent: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._loop.run_until_complete(
            self._async.queue_user_message(message, intent=intent, context=context)
        )

    def get_ingestion_receipts(
        self, source_file: str, content_hashes: list[str]
    ) -> dict[str, UUID]:
        return self._loop.run_until_complete(
            self._async.get_ingestion_receipts(source_file, content_hashes)
        )

    def record_ingestion_receipts(self, items: list[dict[str, Any]]) -> int:
        return self._loop.run_until_complete(
            self._async.record_ingestion_receipts(items)
        )


def hydrated_context_to_render_json(context: HydratedContext) -> dict[str, Any]:
    """Serialize a HydratedContext into the JSON shape render_chat_memory_context
    (db/39) consumes. The DB owns the rendered text; Python only ships data."""

    def mem(m: Memory) -> dict[str, Any]:
        out = {
            "content": m.content,
            "tier": m.tier,
            "similarity": m.similarity,
            "trust_level": m.trust_level,
            "source_attribution": m.source_attribution,
            "strength": m.strength,
            "fidelity": m.fidelity,
            "emotional_intensity": m.emotional_intensity,
            "confidence": m.confidence,
        }
        return {k: v for k, v in out.items() if v is not None}

    return {
        "memories": [mem(m) for m in context.memories],
        "partial_activations": [
            {"cluster_name": pa.cluster_name, "keywords": pa.keywords}
            for pa in context.partial_activations
        ],
        "identity": context.identity,
        "worldview": context.worldview,
        "user_model": context.user_model,
        "emotional_state": context.emotional_state,
        "goals": context.goals,
        "urgent_drives": context.urgent_drives,
        "subgraph": context.subgraph,
        "context_paths": context.context_paths,
    }


async def render_chat_memory_context_db(
    conn: asyncpg.Connection,
    context: HydratedContext,
    *,
    max_memories: int = 5,
    max_partials: int = 3,
) -> str:
    """Render the chat memory-context block via the DB-owned renderer.

    render_chat_memory_context (db/39) is the single source of the prompt
    text, including recall hedges, felt-emotion cues (config thresholds), and
    the knowledge-subgraph section. The former Python renderer was deleted;
    golden fixtures in tests/fixtures/prompt_render/ pin the output.
    """
    render_payload = hydrated_context_to_render_json(context)
    raw = await conn.fetchval(
        "SELECT render_chat_memory_context($1::jsonb, $2::int, $3::int)",
        json.dumps(render_payload, default=str),
        max_memories,
        max_partials,
    )
    text = str(raw or "")
    if context.user_model and "## User Model" not in text:
        user_model_raw = await conn.fetchval(
            "SELECT render_user_model_context($1::jsonb)",
            json.dumps(render_payload.get("user_model") or [], default=str),
        )
        user_model_text = str(user_model_raw or "").strip()
        if user_model_text:
            text = (
                f"{text.rstrip()}\n{user_model_text}"
                if text.strip()
                else user_model_text
            )
    return text


def _coerce_json(val: Any) -> Any:
    if isinstance(val, str):
        import json

        return json.loads(val)
    return val
