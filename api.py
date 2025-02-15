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