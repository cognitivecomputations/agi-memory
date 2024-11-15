import pytest

import asyncpg
import json
from typing import Dict


@pytest.fixture
async def db_pool():
    """Create a connection pool for testing"""
    pool = await asyncpg.create_pool(
        "postgresql://agi_user:agi_password@localhost:5432/agi_db"
    )
    yield pool
    await pool.close()


@pytest.fixture(autouse=True)
async def cleanup_after_test(db_pool):
    """Cleanup after each test"""
    yield
    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute("SET search_path = ag_catalog, public;")
        
        # Clear graph
        await conn.execute("""
            SELECT * FROM cypher('memory_graph', $$
                MATCH (n) DETACH DELETE n
            $$) as (result ag_catalog.agtype)
        """)
        
        # Clear relational tables
        await conn.execute("""
            TRUNCATE TABLE 
                episodic_memories,
                semantic_memories,
                procedural_memories,
                strategic_memories,
                memory_changes,
                memories
            CASCADE
        """)


async def test_extensions(db_pool):
    """Test that required PostgreSQL extensions are installed and properly loaded"""
    async with db_pool.acquire() as conn:
        # Check extensions exist
        extensions = await conn.fetch("""
            SELECT extname, extversion FROM pg_extension
        """)
        ext_names = {ext["extname"]: ext["extversion"] for ext in extensions}

        assert "vector" in ext_names, "pgvector extension not installed"
        assert "age" in ext_names, "AGE extension not installed"
        assert "btree_gist" in ext_names, "btree_gist extension not installed"
        assert "pg_trgm" in ext_names, "pg_trgm extension not installed"
        assert "cube" in ext_names, "cube extension not installed"

        # Verify AGE is loaded
        await conn.execute("LOAD 'age';")
        await conn.execute("SET search_path = ag_catalog, public;")
        result = await conn.fetchval("""
            SELECT count(*) FROM ag_catalog.ag_graph
        """)
        assert result >= 0, "AGE extension not properly loaded"


async def test_memory_tables(db_pool):
    """Test that all memory tables exist with correct columns and constraints"""
    async with db_pool.acquire() as conn:
        # First check if tables exist
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        table_names = {t['table_name'] for t in tables}
        
        assert 'working_memory' in table_names, "working_memory table not found"
        assert 'memories' in table_names, "memories table not found"
        assert 'episodic_memories' in table_names, "episodic_memories table not found"
        
        # Then check columns
        memories = await conn.fetch("""
            SELECT column_name, data_type, is_nullable 
            FROM information_schema.columns 
            WHERE table_name = 'memories'
        """)
        columns = {col["column_name"]: col for col in memories}

        assert "relevance_score" in columns, "relevance_score column not found"
        assert "last_accessed" in columns, "last_accessed column not found"
        assert "id" in columns and columns["id"]["data_type"] == "uuid"
        assert "content" in columns and columns["content"]["is_nullable"] == "NO"
        assert "embedding" in columns
        assert "type" in columns


async def test_memory_storage(db_pool):
    """Test storing and retrieving different types of memories"""
    async with db_pool.acquire() as conn:
        # Test each memory type
        memory_types = ['episodic', 'semantic', 'procedural', 'strategic']
        
        for mem_type in memory_types:
            # Cast the type explicitly
            memory_id = await conn.fetchval("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding
                ) VALUES (
                    $1::memory_type,
                    'Test ' || $1 || ' memory',
                    array_fill(0, ARRAY[1536])::vector
                ) RETURNING id
            """, mem_type)

            assert memory_id is not None

            # Store type-specific details
            if mem_type == 'episodic':
                await conn.execute("""
                    INSERT INTO episodic_memories (
                        memory_id,
                        action_taken,
                        context,
                        result,
                        emotional_valence
                    ) VALUES ($1, $2, $3, $4, 0.5)
                """, 
                    memory_id,
                    json.dumps({"action": "test"}),
                    json.dumps({"context": "test"}),
                    json.dumps({"result": "success"})
                )
            # Add other memory type tests...

        # Verify storage and relationships
        for mem_type in memory_types:
            count = await conn.fetchval("""
                SELECT COUNT(*) 
                FROM memories m 
                WHERE m.type = $1
            """, mem_type)
            assert count > 0, f"No {mem_type} memories stored"


async def test_memory_importance(db_pool):
    """Test memory importance updating"""
    async with db_pool.acquire() as conn:
        # Create test memory
        memory_id = await conn.fetchval(
            """
            INSERT INTO memories (
                type, 
                content, 
                embedding,
                importance,
                access_count
            ) VALUES (
                'semantic',
                'Important test content',
                array_fill(0, ARRAY[1536])::vector,
                0.5,
                0
            ) RETURNING id
        """
        )

        # Update access count to trigger importance recalculation
        await conn.execute(
            """
            UPDATE memories 
            SET access_count = access_count + 1
            WHERE id = $1
        """,
            memory_id,
        )

        # Check that importance was updated
        new_importance = await conn.fetchval(
            """
            SELECT importance 
            FROM memories 
            WHERE id = $1
        """,
            memory_id,
        )

        assert new_importance != 0.5, "Importance should have been updated"


async def test_age_setup(db_pool):
    """Test AGE graph functionality"""
    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute("SET search_path = ag_catalog, public;")
        
        # Test graph exists
        result = await conn.fetch("""
            SELECT * FROM ag_catalog.ag_graph
            WHERE name = 'memory_graph'::name
        """)
        assert len(result) == 1, "memory_graph not found"
        
        # Test vertex label
        result = await conn.fetch("""
            SELECT * FROM ag_catalog.ag_label
            WHERE name = 'MemoryNode'::name 
            AND graph = (
                SELECT graphid FROM ag_catalog.ag_graph 
                WHERE name = 'memory_graph'::name
            )
        """)
        assert len(result) == 1, "MemoryNode label not found"
        
        # Test creating and querying nodes
        await conn.execute("""
            SELECT * FROM ag_catalog.cypher(
                'memory_graph',
                $$CREATE (n:MemoryNode {test: true}) RETURN n$$
            ) as (result ag_catalog.agtype);
        """)
        
        # Query with proper return type
        result = await conn.fetch("""
            SELECT * FROM ag_catalog.cypher(
                'memory_graph',
                $$MATCH (n:MemoryNode {test: true}) RETURN count(n)$$
            ) as (count ag_catalog.agtype);
        """)
        count = int(result[0]['count'])
        assert count > 0, "Failed to create and query nodes"


async def test_memory_relationships(db_pool):
    """Test graph relationships between different memory types"""
    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute("SET search_path = ag_catalog, public;")
        
        memory_pairs = [
            ('semantic', 'semantic', 'RELATES_TO'),
            ('episodic', 'semantic', 'LEADS_TO'),
            ('procedural', 'strategic', 'IMPLEMENTS')
        ]
        
        for source_type, target_type, rel_type in memory_pairs:
            # Create source and target memories
            source_id = await conn.fetchval("""
                INSERT INTO memories (type, content, embedding)
                VALUES ($1::memory_type, 'Source ' || $1, array_fill(0, ARRAY[1536])::vector)
                RETURNING id
            """, source_type)
            
            target_id = await conn.fetchval("""
                INSERT INTO memories (type, content, embedding)
                VALUES ($1::memory_type, 'Target ' || $1, array_fill(0, ARRAY[1536])::vector)
                RETURNING id
            """, target_type)
            
            # Create nodes and relationship in graph using string formatting for Cypher
            cypher_query = f"""
                SELECT * FROM ag_catalog.cypher(
                    'memory_graph',
                    $$
                    CREATE (a:MemoryNode {{memory_id: '{str(source_id)}', type: '{source_type}'}}),
                           (b:MemoryNode {{memory_id: '{str(target_id)}', type: '{target_type}'}}),
                           (a)-[r:{rel_type}]->(b)
                    RETURN a, r, b
                    $$
                ) as (a ag_catalog.agtype, r ag_catalog.agtype, b ag_catalog.agtype)
            """
            await conn.execute(cypher_query)
            
            # Verify the relationship was created
            verify_query = f"""
                SELECT * FROM ag_catalog.cypher(
                    'memory_graph',
                    $$
                    MATCH (a:MemoryNode)-[r:{rel_type}]->(b:MemoryNode)
                    WHERE a.memory_id = '{str(source_id)}' AND b.memory_id = '{str(target_id)}'
                    RETURN a, r, b
                    $$
                ) as (a ag_catalog.agtype, r ag_catalog.agtype, b ag_catalog.agtype)
            """
            result = await conn.fetch(verify_query)
            assert len(result) > 0, f"Relationship {rel_type} not found"


async def test_memory_type_specifics(db_pool):
    """Test type-specific memory storage and constraints"""
    async with db_pool.acquire() as conn:
        # Test semantic memory with confidence
        semantic_id = await conn.fetchval("""
            WITH mem AS (
                INSERT INTO memories (type, content, embedding)
                VALUES ('semantic'::memory_type, 'Test fact', array_fill(0, ARRAY[1536])::vector)
                RETURNING id
            )
            INSERT INTO semantic_memories (memory_id, confidence, category)
            SELECT id, 0.85, ARRAY['test']
            FROM mem
            RETURNING memory_id
        """)
        
        # Test procedural memory success rate calculation
        procedural_id = await conn.fetchval("""
            WITH mem AS (
                INSERT INTO memories (type, content, embedding)
                VALUES ('procedural'::memory_type, 'Test procedure', array_fill(0, ARRAY[1536])::vector)
                RETURNING id
            )
            INSERT INTO procedural_memories (
                memory_id, 
                steps,
                success_count,
                total_attempts
            )
            SELECT id, 
                   '{"steps": ["step1", "step2"]}'::jsonb,
                   8,
                   10
            FROM mem
            RETURNING memory_id
        """)
        
        # Verify success rate calculation
        success_rate = await conn.fetchval("""
            SELECT success_rate 
            FROM procedural_memories 
            WHERE memory_id = $1
        """, procedural_id)
        
        assert success_rate == 0.8, "Success rate calculation incorrect"


async def test_memory_status_transitions(db_pool):
    """Test memory status transitions and tracking"""
    async with db_pool.acquire() as conn:
        # First create trigger if it doesn't exist
        await conn.execute("""
            CREATE OR REPLACE FUNCTION track_memory_changes()
            RETURNS TRIGGER AS $$
            BEGIN
                INSERT INTO memory_changes (
                    memory_id,
                    change_type,
                    old_value,
                    new_value
                ) VALUES (
                    NEW.id,
                    'status_change',
                    jsonb_build_object('status', OLD.status),
                    jsonb_build_object('status', NEW.status)
                );
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;

            DROP TRIGGER IF EXISTS track_status_changes ON memories;
            CREATE TRIGGER track_status_changes
                AFTER UPDATE OF status ON memories
                FOR EACH ROW
                EXECUTE FUNCTION track_memory_changes();
        """)

        # Create test memory
        memory_id = await conn.fetchval("""
            INSERT INTO memories (type, content, embedding, status)
            VALUES (
                'semantic'::memory_type,
                'Test content',
                array_fill(0, ARRAY[1536])::vector,
                'active'::memory_status
            ) RETURNING id
        """)

        # Archive memory and verify change tracking
        await conn.execute("""
            UPDATE memories 
            SET status = 'archived'::memory_status
            WHERE id = $1
        """, memory_id)

        changes = await conn.fetch("""
            SELECT * FROM memory_changes
            WHERE memory_id = $1
            ORDER BY changed_at DESC
        """, memory_id)

        assert len(changes) > 0, "Status change not tracked"


async def test_vector_search(db_pool):
    """Test vector similarity search"""
    async with db_pool.acquire() as conn:
        # Create test memories with different embeddings
        test_embeddings = [
            '[' + ','.join(['1.0'] * 1536) + ']',
            '[' + ','.join(['0.5'] * 1536) + ']',
            '[' + ','.join(['0.0'] * 1536) + ']'
        ]
        
        for emb in test_embeddings:
            await conn.execute("""
                INSERT INTO memories (type, content, embedding)
                VALUES (
                    'semantic'::memory_type,
                    'Test content',
                    $1::vector
                )
            """, emb)
        
        # Test nearest neighbor search
        results = await conn.fetch("""
            SELECT id, 1 - (embedding <-> $1::vector) as similarity
            FROM memories
            ORDER BY embedding <-> $1::vector
            LIMIT 2
        """, test_embeddings[0])
        
        assert len(results) == 2, "Wrong number of results"
        assert results[0]['similarity'] > results[1]['similarity'], "Incorrect similarity ordering"


async def test_complex_graph_queries(db_pool):
    """Test more complex graph operations and queries"""
    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute("SET search_path = ag_catalog, public;")
        
        # Create a chain of related memories
        memory_chain = [
            ('episodic', 'Start event'),
            ('semantic', 'Derived knowledge'),
            ('procedural', 'Applied procedure')
        ]
        
        prev_id = None
        for mem_type, content in memory_chain:
            # Create memory
            curr_id = await conn.fetchval("""
                INSERT INTO memories (type, content, embedding)
                VALUES ($1::memory_type, $2, array_fill(0, ARRAY[1536])::vector)
                RETURNING id
            """, mem_type, content)
            
            # Create graph node
            await conn.execute(f"""
                SELECT * FROM cypher('memory_graph', $$
                    CREATE (n:MemoryNode {{
                        memory_id: '{curr_id}',
                        type: '{mem_type}'
                    }})
                    RETURN n
                $$) as (n ag_catalog.agtype)
            """)
            
            if prev_id:
                await conn.execute(f"""
                    SELECT * FROM cypher('memory_graph', $$
                        MATCH (a:MemoryNode {{memory_id: '{prev_id}'}}),
                              (b:MemoryNode {{memory_id: '{curr_id}'}})
                        CREATE (a)-[r:LEADS_TO]->(b)
                        RETURN r
                    $$) as (r ag_catalog.agtype)
                """)
            
            prev_id = curr_id
        
        # Test path query with fixed syntax
        result = await conn.fetch("""
            SELECT * FROM cypher('memory_graph', $$
                MATCH p = (s:MemoryNode)-[*]->(t:MemoryNode)
                WHERE s.type = 'episodic' AND t.type = 'procedural'
                RETURN p
            $$) as (path ag_catalog.agtype)
        """)
        
        assert len(result) > 0, "No valid paths found"


async def test_memory_storage_episodic(db_pool):
    """Test storing and retrieving episodic memories"""
    async with db_pool.acquire() as conn:
        # Create base memory
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding,
                importance,
                decay_rate
            ) VALUES (
                'episodic'::memory_type,
                'Test episodic memory',
                array_fill(0, ARRAY[1536])::vector,
                0.5,
                0.01
            ) RETURNING id
        """)

        assert memory_id is not None

        # Store episodic details
        await conn.execute("""
            INSERT INTO episodic_memories (
                memory_id,
                action_taken,
                context,
                result,
                emotional_valence,
                verification_status,
                event_time
            ) VALUES ($1, $2, $3, $4, 0.5, true, CURRENT_TIMESTAMP)
        """, 
            memory_id,
            json.dumps({"action": "test"}),
            json.dumps({"context": "test"}),
            json.dumps({"result": "success"})
        )

        # Verify storage
        count = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM memories m 
            JOIN episodic_memories e ON m.id = e.memory_id
            WHERE m.type = 'episodic'
        """)
        assert count > 0, "No episodic memories stored"


async def test_memory_storage_semantic(db_pool):
    """Test storing and retrieving semantic memories"""
    async with db_pool.acquire() as conn:
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding,
                importance,
                decay_rate
            ) VALUES (
                'semantic'::memory_type,
                'Test semantic memory',
                array_fill(0, ARRAY[1536])::vector,
                0.5,
                0.01
            ) RETURNING id
        """)

        assert memory_id is not None

        await conn.execute("""
            INSERT INTO semantic_memories (
                memory_id,
                confidence,
                source_references,
                contradictions,
                category,
                related_concepts
            ) VALUES ($1, 0.8, $2, $3, $4, $5)
        """,
            memory_id,
            json.dumps({"source": "test"}),
            json.dumps({"contradictions": []}),
            ["test_category"],
            ["test_concept"]
        )

        count = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM memories m 
            JOIN semantic_memories s ON m.id = s.memory_id
            WHERE m.type = 'semantic'
        """)
        assert count > 0, "No semantic memories stored"


async def test_memory_storage_strategic(db_pool):
    """Test storing and retrieving strategic memories"""
    async with db_pool.acquire() as conn:
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding,
                importance,
                decay_rate
            ) VALUES (
                'strategic'::memory_type,
                'Test strategic memory',
                array_fill(0, ARRAY[1536])::vector,
                0.5,
                0.01
            ) RETURNING id
        """)

        assert memory_id is not None

        await conn.execute("""
            INSERT INTO strategic_memories (
                memory_id,
                pattern_description,
                supporting_evidence,
                confidence_score,
                success_metrics,
                adaptation_history,
                context_applicability
            ) VALUES ($1, 'Test pattern', $2, 0.7, $3, $4, $5)
        """,
            memory_id,
            json.dumps({"evidence": ["test"]}),
            json.dumps({"metrics": {"success": 0.8}}),
            json.dumps({"adaptations": []}),
            json.dumps({"contexts": ["test_context"]})
        )

        count = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM memories m 
            JOIN strategic_memories s ON m.id = s.memory_id
            WHERE m.type = 'strategic'
        """)
        assert count > 0, "No strategic memories stored"


async def test_memory_storage_procedural(db_pool):
    """Test storing and retrieving procedural memories"""
    async with db_pool.acquire() as conn:
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding,
                importance,
                decay_rate
            ) VALUES (
                'procedural'::memory_type,
                'Test procedural memory',
                array_fill(0, ARRAY[1536])::vector,
                0.5,
                0.01
            ) RETURNING id
        """)

        assert memory_id is not None

        await conn.execute("""
            INSERT INTO procedural_memories (
                memory_id,
                steps,
                prerequisites,
                success_count,
                total_attempts,
                average_duration,
                failure_points
            ) VALUES ($1, $2, $3, 5, 10, '1 hour', $4)
        """,
            memory_id,
            json.dumps({"steps": ["step1", "step2"]}),
            json.dumps({"prereqs": ["prereq1"]}),
            json.dumps({"failures": []})
        )

        count = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM memories m 
            JOIN procedural_memories p ON m.id = p.memory_id
            WHERE m.type = 'procedural'
        """)
        assert count > 0, "No procedural memories stored"
        
async def test_working_memory(db_pool):
    """Test working memory operations"""
    async with db_pool.acquire() as conn:
        # Test inserting into working memory
        working_memory_id = await conn.fetchval("""
            INSERT INTO working_memory (
                content,
                embedding,
                expiry
            ) VALUES (
                'Test working memory',
                array_fill(0, ARRAY[1536])::vector,
                CURRENT_TIMESTAMP + interval '1 hour'
            ) RETURNING id
        """)
        
        assert working_memory_id is not None, "Failed to insert working memory"
        
        # Test expiry
        expired_count = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM working_memory 
            WHERE expiry < CURRENT_TIMESTAMP
        """)
        
        assert isinstance(expired_count, int), "Failed to query expired memories"

async def test_memory_relevance(db_pool):
    """Test memory relevance score calculation"""
    async with db_pool.acquire() as conn:
        # Create test memory with known values
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding,
                importance,
                decay_rate,
                created_at
            ) VALUES (
                'semantic'::memory_type,
                'Test relevance',
                array_fill(0, ARRAY[1536])::vector,
                0.8,
                0.01,
                CURRENT_TIMESTAMP - interval '1 day'
            ) RETURNING id
        """)
        
        # Check relevance score
        relevance = await conn.fetchval("""
            SELECT relevance_score
            FROM memories
            WHERE id = $1
        """, memory_id)
        
        assert relevance is not None, "Relevance score not calculated"
        assert relevance < 0.8, "Relevance should be less than importance due to decay"
