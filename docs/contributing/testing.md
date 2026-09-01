<!--
title: Testing
summary: Test conventions, running tests, and writing new tests
read_when:
  - "You want to run the test suite"
  - "You want to write new tests"
section: contributing
-->

# Testing

Test conventions, running the suite, and writing new tests.

## Running Tests

```bash
# Ensure services are running
hexis up
hexis doctor

# Run all tests
POSTGRES_HOST=127.0.0.1 pytest tests -q

# Run specific test suites
pytest tests/db -q         # Database integration tests
pytest tests/core -q       # Core API tests
pytest tests/services -q   # Service-level tests
pytest tests/cli -q        # CLI smoke tests
```

Use `POSTGRES_HOST=127.0.0.1` to avoid SSL negotiation flakes when connecting to the local Docker Postgres.

## Public Memory Benchmark

The ordinary suite verifies benchmark validation, scoring, external-command
isolation, stable reference baselines, and a rollback-only journey through the real
Hexis memory functions:

```bash
pytest tests/evals/test_memory_benchmark.py -q
python -m evals.memory_benchmark.run validate
```

A published Hexis score additionally needs the configured embedding service and
five live contradiction-classifier calls:

```bash
python -m evals.memory_benchmark.run run --adapter hexis --live-contradictions
```

Do not add that live-provider run to CI. The public corpus is hash-pinned; change its
version and expected hash deliberately when revising any case or gold label. See
[Public Memory Benchmark](../guides/memory-benchmark.md) for the protocol and
submission disclosure requirements.

## CI

GitHub Actions runs the required `all-checks-pass` gate. The integration lane
starts the prebuilt Hexis brain image and uses `ops/ci/fake_embeddings.py`
instead of the real embedding model, so CI does not download embedding models. The
`migration-survivor` lane seeds an existing database, runs `hexis migrate`'s
underlying migration runner, and verifies the data survives.

## Test Organization

```
tests/
├── db/          # Database schema, functions, triggers
├── core/        # Core API and tool tests
├── services/    # Service orchestration tests
└── cli/         # CLI smoke tests
```

## Conventions

### Framework

- `pytest` + `pytest-asyncio` with session loop scope
- Integration tests using transactions/rollbacks

### Async Tests

All async tests using the `db_pool` fixture must use `loop_scope="session"`:

```python
pytestmark = [pytest.mark.asyncio(loop_scope="session")]
```

### Unique Test Data

Use `get_test_identifier()` from `tests/utils.py` for unique data:

```python
from tests.utils import get_test_identifier

async def test_something(db_pool):
    identifier = get_test_identifier()
    # Use identifier for unique content
```

### Seeding Memories

The `memories` table has a NOT NULL constraint on `embedding`. Use `array_fill` for dummy vectors:

```python
await conn.fetchval("""
    INSERT INTO memories (type, content, embedding, importance, trust_level, status)
    VALUES ('semantic', $1, array_fill(0.1, ARRAY[embedding_dimension()])::vector, 0.8, 0.9, 'active')
    RETURNING id
""", content)
```

### Naming

- Functions: `test_*`
- Files: `test_*.py`
- Descriptive names that explain what's being tested

## Test Coverage Areas

| Area | Priority | Tests |
|------|----------|-------|
| Memory creation/retrieval | High | fast_recall, search_similar, create_* |
| Episodes and neighborhoods | High | Auto-assignment, staleness triggers |
| Concept hierarchy | High | link_memory_to_concept, ancestors |
| Graph operations | Medium | Edge types, SelfNode |
| Maintenance functions | Medium | Cleanup, neighborhood recomputation |
| Identity/worldview | Medium | Confidence updates, relationship edges |
| Views | Low | memory_health, cluster_insights |
| Indexes | Low | HNSW, GIN performance |

## Writing New Tests

1. Place tests in the appropriate directory (`tests/db/`, `tests/core/`, etc.)
2. Use the `db_pool` fixture for database access
3. Use transactions with rollback for isolation
4. Follow existing naming patterns
5. Include both positive and negative test cases

## Related

- [Contributing](index.md) -- development setup and coding style
- [Database](../operations/database.md) -- schema management for test setup
