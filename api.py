import asyncpg
import os

# Global variable to store the connection pool
_pool = None

async def init_db():
    """Initializes the database connection pool."""
    global _pool
    _pool = await asyncpg.create_pool(
        user=os.environ.get("POSTGRES_USER", "agi_user"),
        password=os.environ.get("POSTGRES_PASSWORD", "agi_password!QAZ"),
        database=os.environ.get("POSTGRES_DB", "agi_db"),
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", 5432))
    )
    return _pool

async def close_db():
    """Closes the database connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None

async def get_db() -> asyncpg.Pool:
    """Returns the database connection pool.
       Raises an exception if the pool has not been initialized.
    """
    if _pool is None:
        raise Exception("Database connection pool not initialized. Call init_db() first.")
    return _pool


import json

async def create_memory(memory_type: str, content: str, embedding: list[float], context: dict | None = None, action_taken: dict | None = None, result: dict | None = None, emotional_valence: float | None = None) -> str:
    """Creates a new memory in the database.

    Args:
        memory_type: The type of memory ('episodic', 'semantic', 'procedural', 'strategic').
        content: The content of the memory.
        embedding: The memory's embedding vector.
        context: Optional context for episodic memories.
        action_taken: Optional action taken for episodic memories.
        result: Optional result for episodic memories.
        emotional_valence: Optional emotional valence for episodic memories.

    Returns:
        The ID of the newly created memory.
    """
    pool = await get_db()
    async with pool.acquire() as conn:
        async with conn.transaction():
            # Insert into the main memories table
            memory_id = await conn.fetchval(
                """
                INSERT INTO memories (type, content, embedding)
                VALUES ($1::memory_type, $2, $3)
                RETURNING id
                """,
                memory_type,
                content,
                embedding,
            )

            # Insert into type-specific tables
            if memory_type == 'episodic':
                await conn.execute(
                    """
                    INSERT INTO episodic_memories (memory_id, context, action_taken, result, emotional_valence)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    memory_id,
                    json.dumps(context) if context else None,
                    json.dumps(action_taken) if action_taken else None,
                    json.dumps(result) if result else None,
                    emotional_valence
                )
            elif memory_type == 'semantic':
                # TODO: Implement semantic memory creation
                pass
            elif memory_type == 'procedural':
                # TODO: Implement procedural memory creation
                pass
            elif memory_type == 'strategic':
                # TODO: Implement strategic memory creation
                pass
            else:
                raise ValueError(f"Invalid memory type: {memory_type}")


            return str(memory_id)