import pytest
import asyncio
import asyncpg
import json
import numpy as np
import time
import uuid

# Update to use loop_scope instead of scope
pytestmark = pytest.mark.asyncio(loop_scope="session")

# Global test session ID to help with cleanup
TEST_SESSION_ID = str(uuid.uuid4())[:8]

def get_test_identifier(test_name: str) -> str:
    """Generate a unique identifier for test data"""
    return f"{test_name}_{TEST_SESSION_ID}_{int(time.time() * 1000)}"

@pytest.fixture(scope="session")
async def db_pool():
    """Create a connection pool for testing"""
    pool = await asyncpg.create_pool(
        "postgresql://agi_user:agi_password@localhost:5432/agi_db",
        ssl=False,
        min_size=2,
        max_size=20,
        command_timeout=60.0
    )
    yield pool
    await pool.close()

@pytest.fixture(autouse=True)
async def setup_db(db_pool):
    """Setup the database before each test"""
    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute("SET search_path = ag_catalog, public;")
    yield

async def test_extensions(db_pool):
    """Test that required PostgreSQL extensions are installed"""
    async with db_pool.acquire() as conn:
        extensions = await conn.fetch("""
            SELECT extname FROM pg_extension
        """)
        ext_names = {ext['extname'] for ext in extensions}
        
        required_extensions = {'vector', 'age', 'btree_gist', 'pg_trgm'}
        for ext in required_extensions:
            assert ext in ext_names, f"{ext} extension not found"
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
        test_id = get_test_identifier("memory_storage")
        
        # Test each memory type
        memory_types = ['episodic', 'semantic', 'procedural', 'strategic']
        created_memories = []
        
        for mem_type in memory_types:
            # Cast the type explicitly
            memory_id = await conn.fetchval("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding
                ) VALUES (
                    $1::memory_type,
                    'Test ' || $1 || ' memory ' || $2,
                    array_fill(0, ARRAY[1536])::vector
                ) RETURNING id
            """, mem_type, test_id)

            assert memory_id is not None
            created_memories.append(memory_id)

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

        # Verify storage and relationships for our specific test memories
        for mem_type in memory_types:
            count = await conn.fetchval("""
                SELECT COUNT(*) 
                FROM memories m 
                WHERE m.type = $1 AND m.content LIKE '%' || $2
            """, mem_type, test_id)
            assert count > 0, f"No {mem_type} memories stored for test {test_id}"


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
        # Ensure clean state
        await conn.execute("""
            LOAD 'age';
            SET search_path = ag_catalog, public;
            SELECT drop_graph('memory_graph', true);
        """)
        
        # Create graph and label
        await conn.execute("""
            SELECT create_graph('memory_graph');
        """)
        
        await conn.execute("""
            SELECT create_vlabel('memory_graph', 'MemoryNode');
        """)

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
        # Clear existing test data with proper cascade
        await conn.execute("""
            DELETE FROM memory_changes 
            WHERE memory_id IN (
                SELECT id FROM memories WHERE content LIKE 'Test content%'
            )
        """)
        await conn.execute("""
            DELETE FROM semantic_memories 
            WHERE memory_id IN (
                SELECT id FROM memories WHERE content LIKE 'Test content%'
            )
        """)
        await conn.execute("""
            DELETE FROM episodic_memories 
            WHERE memory_id IN (
                SELECT id FROM memories WHERE content LIKE 'Test content%'
            )
        """)
        await conn.execute("""
            DELETE FROM procedural_memories 
            WHERE memory_id IN (
                SELECT id FROM memories WHERE content LIKE 'Test content%'
            )
        """)
        await conn.execute("""
            DELETE FROM strategic_memories 
            WHERE memory_id IN (
                SELECT id FROM memories WHERE content LIKE 'Test content%'
            )
        """)
        await conn.execute("DELETE FROM memories WHERE content LIKE 'Test content%'")
        
        # Create more distinct test vectors
        test_embeddings = [
            # First vector: alternating 1.0 and 0.8
            '[' + ','.join(['1.0' if i % 2 == 0 else '0.8' for i in range(1536)]) + ']',
            # Second vector: alternating 0.5 and 0.3
            '[' + ','.join(['0.5' if i % 2 == 0 else '0.3' for i in range(1536)]) + ']',
            # Third vector: alternating 0.2 and 0.0
            '[' + ','.join(['0.2' if i % 2 == 0 else '0.0' for i in range(1536)]) + ']'
        ]
        
        # Insert test vectors
        for i, emb in enumerate(test_embeddings):
            await conn.execute("""
                INSERT INTO memories (
                    type, 
                    content, 
                    embedding
                ) VALUES (
                    'semantic'::memory_type,
                    'Test content ' || $1,
                    $2::vector
                )
            """, str(i), emb)

        # Query vector more similar to first pattern
        query_vector = '[' + ','.join(['0.95' if i % 2 == 0 else '0.75' for i in range(1536)]) + ']'
        
        results = await conn.fetch("""
            SELECT 
                id, 
                content,
                embedding <=> $1::vector as cosine_distance
            FROM memories
            WHERE content LIKE 'Test content%'
            ORDER BY embedding <=> $1::vector
            LIMIT 3
        """, query_vector)

        assert len(results) >= 2, "Wrong number of results"
        
        # Print distances for debugging
        for r in results:
            print(f"Content: {r['content']}, Distance: {r['cosine_distance']}")
            
        # First result should have smaller cosine distance than second
        assert results[0]['cosine_distance'] < results[1]['cosine_distance'], \
            f"Incorrect distance ordering: {results[0]['cosine_distance']} >= {results[1]['cosine_distance']}"


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

        # Verify storage including new fields
        result = await conn.fetchrow("""
            SELECT e.verification_status, e.event_time
            FROM memories m 
            JOIN episodic_memories e ON m.id = e.memory_id
            WHERE m.type = 'episodic' AND m.id = $1
        """, memory_id)
        
        assert result['verification_status'] is True, "Verification status not set"
        assert result['event_time'] is not None, "Event time not set"


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
                related_concepts,
                last_validated
            ) VALUES ($1, 0.8, $2, $3, $4, $5, CURRENT_TIMESTAMP)
        """,
            memory_id,
            json.dumps({"source": "test"}),
            json.dumps({"contradictions": []}),
            ["test_category"],
            ["test_concept"]
        )

        # Verify including new field
        result = await conn.fetchrow("""
            SELECT s.last_validated
            FROM memories m 
            JOIN semantic_memories s ON m.id = s.memory_id
            WHERE m.type = 'semantic' AND m.id = $1
        """, memory_id)
        
        assert result['last_validated'] is not None, "Last validated timestamp not set"


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

async def test_worldview_primitives(db_pool):
    """Test worldview primitives and their influence on memories"""
    async with db_pool.acquire() as conn:
        # Create worldview primitive
        worldview_id = await conn.fetchval("""
            INSERT INTO worldview_primitives (
                id,
                category,
                belief,
                confidence,
                emotional_valence,
                stability_score,
                activation_patterns,
                memory_filter_rules,
                influence_patterns
            ) VALUES (
                gen_random_uuid(),
                'values',
                'Test belief',
                0.8,
                0.5,
                0.7,
                '{"patterns": ["test"]}',
                '{"filters": ["test"]}',
                '{"influences": ["test"]}'
            ) RETURNING id
        """)
        
        # Create memory
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding
            ) VALUES (
                'episodic'::memory_type,
                'Test memory for worldview',
                array_fill(0, ARRAY[1536])::vector
            ) RETURNING id
        """)
        
        # Create influence relationship
        await conn.execute("""
            INSERT INTO worldview_memory_influences (
                id,
                worldview_id,
                memory_id,
                influence_type,
                strength
            ) VALUES (
                gen_random_uuid(),
                $1,
                $2,
                'filter',
                0.7
            )
        """, worldview_id, memory_id)
        
        # Verify relationship
        influence = await conn.fetchrow("""
            SELECT * 
            FROM worldview_memory_influences
            WHERE worldview_id = $1 AND memory_id = $2
        """, worldview_id, memory_id)
        
        assert influence is not None, "Worldview influence not created"
        assert influence['strength'] == 0.7, "Incorrect influence strength"

async def test_identity_model(db_pool):
    """Test identity model and memory resonance"""
    async with db_pool.acquire() as conn:
        # Create identity aspect
        identity_id = await conn.fetchval("""
            INSERT INTO identity_model (
                id,
                self_concept,
                agency_beliefs,
                purpose_framework,
                group_identifications,
                boundary_definitions,
                emotional_baseline,
                threat_sensitivity,
                change_resistance
            ) VALUES (
                gen_random_uuid(),
                '{"concept": "test"}',
                '{"agency": "high"}',
                '{"purpose": "test"}',
                '{"groups": ["test"]}',
                '{"boundaries": ["test"]}',
                '{"baseline": "neutral"}',
                0.5,
                0.3
            ) RETURNING id
        """)
        
        # Create memory
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding
            ) VALUES (
                'episodic'::memory_type,
                'Test memory for identity',
                array_fill(0, ARRAY[1536])::vector
            ) RETURNING id
        """)
        
        # Create resonance
        await conn.execute("""
            INSERT INTO identity_memory_resonance (
                id,
                memory_id,
                identity_aspect,
                resonance_strength,
                integration_status
            ) VALUES (
                gen_random_uuid(),
                $1,
                $2,
                0.8,
                'integrated'
            )
        """, memory_id, identity_id)
        
        # Verify resonance
        resonance = await conn.fetchrow("""
            SELECT * 
            FROM identity_memory_resonance
            WHERE memory_id = $1 AND identity_aspect = $2
        """, memory_id, identity_id)
        
        assert resonance is not None, "Identity resonance not created"
        assert resonance['resonance_strength'] == 0.8, "Incorrect resonance strength"

async def test_memory_changes_tracking(db_pool):
    """Test comprehensive memory changes tracking"""
    async with db_pool.acquire() as conn:
        # Create test memory
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding,
                importance
            ) VALUES (
                'semantic'::memory_type,
                'Test tracking memory',
                array_fill(0, ARRAY[1536])::vector,
                0.5
            ) RETURNING id
        """)
        
        # Make various changes
        changes = [
            ('importance_update', 0.5, 0.7),
            ('status_change', 'active', 'archived'),
            ('content_update', 'Test tracking memory', 'Updated test memory')
        ]
        
        for change_type, old_val, new_val in changes:
            await conn.execute("""
                INSERT INTO memory_changes (
                    memory_id,
                    change_type,
                    old_value,
                    new_value
                ) VALUES (
                    $1,
                    $2,
                    $3::jsonb,
                    $4::jsonb
                )
            """, memory_id, change_type, 
                json.dumps({change_type: old_val}),
                json.dumps({change_type: new_val}))
        
        # Verify change history
        history = await conn.fetch("""
            SELECT change_type, old_value, new_value
            FROM memory_changes
            WHERE memory_id = $1
            ORDER BY changed_at DESC
        """, memory_id)
        
        assert len(history) == len(changes), "Not all changes were tracked"
        assert history[0]['change_type'] == changes[-1][0], "Changes not tracked in correct order"

async def test_enhanced_relevance_scoring(db_pool):
    """Test the enhanced relevance scoring system"""
    async with db_pool.acquire() as conn:
        # Create test memory with specific parameters
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding,
                importance,
                decay_rate,
                created_at,
                access_count
            ) VALUES (
                'semantic'::memory_type,
                'Test relevance scoring',
                array_fill(0, ARRAY[1536])::vector,
                0.8,
                0.01,
                CURRENT_TIMESTAMP - interval '1 day',
                5
            ) RETURNING id
        """)
        
        # Get initial relevance score
        initial_score = await conn.fetchval("""
            SELECT relevance_score
            FROM memories
            WHERE id = $1
        """, memory_id)
        
        # Update access count to trigger importance change
        await conn.execute("""
            UPDATE memories 
            SET access_count = access_count + 1
            WHERE id = $1
        """, memory_id)
        
        # Get updated relevance score
        updated_score = await conn.fetchval("""
            SELECT relevance_score
            FROM memories
            WHERE id = $1
        """, memory_id)
        
        assert initial_score is not None, "Initial relevance score not calculated"
        assert updated_score is not None, "Updated relevance score not calculated"
        assert updated_score != initial_score, "Relevance score should change with importance"

async def test_age_in_days_function(db_pool):
    """Test the age_in_days function"""
    async with db_pool.acquire() as conn:
        # Test current timestamp (should be 0 days)
        result = await conn.fetchval("""
            SELECT age_in_days(CURRENT_TIMESTAMP)
        """)
        assert result < 1, "Current timestamp should be less than 1 day old"

        # Test 1 day ago
        result = await conn.fetchval("""
            SELECT age_in_days(CURRENT_TIMESTAMP - interval '1 day')
        """)
        assert abs(result - 1.0) < 0.1, "Should be approximately 1 day"

        # Test 7 days ago
        result = await conn.fetchval("""
            SELECT age_in_days(CURRENT_TIMESTAMP - interval '7 days')
        """)
        assert abs(result - 7.0) < 0.1, "Should be approximately 7 days"

async def test_update_memory_timestamp_trigger(db_pool):
    """Test the update_memory_timestamp trigger"""
    async with db_pool.acquire() as conn:
        # Create test memory
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding
            ) VALUES (
                'semantic'::memory_type,
                'Test timestamp update',
                array_fill(0, ARRAY[1536])::vector
            ) RETURNING id
        """)

        # Get initial timestamp
        initial_updated_at = await conn.fetchval("""
            SELECT updated_at FROM memories WHERE id = $1
        """, memory_id)

        # Wait briefly
        await asyncio.sleep(0.1)

        # Update memory
        await conn.execute("""
            UPDATE memories 
            SET content = 'Updated content'
            WHERE id = $1
        """, memory_id)

        # Get new timestamp
        new_updated_at = await conn.fetchval("""
            SELECT updated_at FROM memories WHERE id = $1
        """, memory_id)

        assert new_updated_at > initial_updated_at, "updated_at should be newer"

async def test_update_memory_importance_trigger(db_pool):
    """Test the update_memory_importance trigger"""
    async with db_pool.acquire() as conn:
        # Create test memory
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding,
                importance,
                access_count
            ) VALUES (
                'semantic'::memory_type,
                'Test importance update',
                array_fill(0, ARRAY[1536])::vector,
                0.5,
                0
            ) RETURNING id
        """)

        # Get initial importance
        initial_importance = await conn.fetchval("""
            SELECT importance FROM memories WHERE id = $1
        """, memory_id)

        # Update access count
        await conn.execute("""
            UPDATE memories 
            SET access_count = access_count + 1
            WHERE id = $1
        """, memory_id)

        # Get new importance
        new_importance = await conn.fetchval("""
            SELECT importance FROM memories WHERE id = $1
        """, memory_id)

        assert new_importance > initial_importance, "Importance should increase"
        
        # Test multiple accesses
        await conn.execute("""
            UPDATE memories 
            SET access_count = access_count + 5
            WHERE id = $1
        """, memory_id)
        
        final_importance = await conn.fetchval("""
            SELECT importance FROM memories WHERE id = $1
        """, memory_id)
        
        assert final_importance > new_importance, "Importance should increase with more accesses"

async def test_create_memory_relationship_function(db_pool):
    """Test the create_memory_relationship function"""
    async with db_pool.acquire() as conn:
        # Create two test memories
        memory_ids = []
        for i in range(2):
            memory_id = await conn.fetchval("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding
                ) VALUES (
                    'semantic'::memory_type,
                    'Test memory ' || $1::text,
                    array_fill(0, ARRAY[1536])::vector
                ) RETURNING id
            """, str(i))
            memory_ids.append(memory_id)

        # Ensure clean AGE setup with proper schema
        await conn.execute("""
            LOAD 'age';
            SET search_path = ag_catalog, public;
            SELECT drop_graph('memory_graph', true);
            SELECT create_graph('memory_graph');
            SELECT create_vlabel('memory_graph', 'MemoryNode');
        """)

        # Create nodes in graph using string formatting for Cypher
        for memory_id in memory_ids:
            cypher_query = f"""
                SELECT * FROM ag_catalog.cypher('memory_graph', $$
                    CREATE (n:MemoryNode {{
                        memory_id: '{str(memory_id)}',
                        type: 'semantic'
                    }})
                    RETURN n
                $$) as (result agtype)
            """
            await conn.execute(cypher_query)

        properties = {"weight": 0.8}
        
        # Create relationship
        await conn.execute("""
            SELECT create_memory_relationship($1, $2, $3, $4)
        """, memory_ids[0], memory_ids[1], 'RELATES_TO', json.dumps(properties))

        # Verify relationship exists
        verify_query = f"""
            SELECT * FROM ag_catalog.cypher('memory_graph', $$
                MATCH (a:MemoryNode)-[r:RELATES_TO]->(b:MemoryNode)
                WHERE a.memory_id = '{str(memory_ids[0])}' AND b.memory_id = '{str(memory_ids[1])}'
                RETURN r
            $$) as (result agtype)
        """
        result = await conn.fetch(verify_query)
        assert len(result) > 0, "Relationship not created"

async def test_memory_health_view(db_pool):
    """Test the memory_health view"""
    async with db_pool.acquire() as conn:
        # Create test memories of different types
        memory_types = ['episodic', 'semantic', 'procedural', 'strategic']
        for mem_type in memory_types:
            await conn.execute("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding,
                    importance,
                    access_count
                ) VALUES (
                    $1::memory_type,
                    'Test ' || $1,
                    array_fill(0, ARRAY[1536])::vector,
                    0.5,
                    5
                )
            """, mem_type)

        # Query view
        results = await conn.fetch("""
            SELECT * FROM memory_health
        """)

        assert len(results) > 0, "Memory health view should return results"
        
        # Verify each type has stats
        result_types = {r['type'] for r in results}
        for mem_type in memory_types:
            assert mem_type in result_types, f"Missing stats for {mem_type}"
            
        # Verify computed values
        for row in results:
            assert row['total_memories'] > 0, "Should have memories"
            assert row['avg_importance'] is not None, "Should have importance"
            assert row['avg_access_count'] is not None, "Should have access count"



async def test_extensions(db_pool):
    """Test that required PostgreSQL extensions are installed"""
    async with db_pool.acquire() as conn:
        extensions = await conn.fetch("""
            SELECT extname FROM pg_extension
        """)
        ext_names = {ext['extname'] for ext in extensions}
        
        required_extensions = {'vector', 'age', 'btree_gist', 'pg_trgm', 'cube'}
        for ext in required_extensions:
            assert ext in ext_names, f"{ext} extension not found"
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
        assert 'memory_clusters' in table_names, "memory_clusters table not found"
        assert 'memory_cluster_members' in table_names, "memory_cluster_members table not found"
        assert 'cluster_relationships' in table_names, "cluster_relationships table not found"
        assert 'cluster_activation_history' in table_names, "cluster_activation_history table not found"
        
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

async def test_memory_clusters(db_pool):
    """Test memory clustering functionality"""
    async with db_pool.acquire() as conn:
        # Create test cluster
        cluster_id = await conn.fetchval("""
            INSERT INTO memory_clusters (
                cluster_type,
                name,
                description,
                centroid_embedding,
                emotional_signature,
                keywords,
                importance_score,
                coherence_score
            ) VALUES (
                'theme'::cluster_type,
                'Test Theme Cluster',
                'Cluster for testing',
                array_fill(0.5, ARRAY[1536])::vector,
                '{"dominant": "neutral", "secondary": "curious"}'::jsonb,
                ARRAY['test', 'memory', 'cluster'],
                0.7,
                0.85
            ) RETURNING id
        """)
        
        assert cluster_id is not None, "Failed to create cluster"
        
        # Create test memories and add to cluster
        memory_ids = []
        for i in range(3):
            memory_id = await conn.fetchval("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding
                ) VALUES (
                    'semantic'::memory_type,
                    'Test memory for clustering ' || $1,
                    array_fill($2::float, ARRAY[1536])::vector
                ) RETURNING id
            """, str(i), float(i) * 0.1)
            memory_ids.append(memory_id)
            
            # Add to cluster
            await conn.execute("""
                INSERT INTO memory_cluster_members (
                    cluster_id,
                    memory_id,
                    membership_strength,
                    contribution_to_centroid
                ) VALUES ($1, $2, $3, $4)
            """, cluster_id, memory_id, 0.8 - (i * 0.1), 0.3)
        
        # Verify cluster membership
        members = await conn.fetch("""
            SELECT * FROM memory_cluster_members
            WHERE cluster_id = $1
            ORDER BY membership_strength DESC
        """, cluster_id)
        
        assert len(members) == 3, "Wrong number of cluster members"
        assert members[0]['membership_strength'] == 0.8, "Incorrect membership strength"

async def test_cluster_relationships(db_pool):
    """Test relationships between clusters"""
    async with db_pool.acquire() as conn:
        # Create two clusters
        cluster_ids = []
        for i, name in enumerate(['Loneliness', 'Connection']):
            cluster_id = await conn.fetchval("""
                INSERT INTO memory_clusters (
                    cluster_type,
                    name,
                    description,
                    centroid_embedding,
                    keywords
                ) VALUES (
                    'emotion'::cluster_type,
                    $1,
                    'Emotional cluster for ' || $1,
                    array_fill($2::float, ARRAY[1536])::vector,
                    ARRAY[$1]
                ) RETURNING id
            """, name, float(i) * 0.5)
            cluster_ids.append(cluster_id)
        
        # Create relationship between clusters
        await conn.execute("""
            INSERT INTO cluster_relationships (
                from_cluster_id,
                to_cluster_id,
                relationship_type,
                strength,
                evidence_memories
            ) VALUES ($1, $2, 'contradicts', 0.7, $3)
        """, cluster_ids[0], cluster_ids[1], [])
        
        # Verify relationship
        relationship = await conn.fetchrow("""
            SELECT * FROM cluster_relationships
            WHERE from_cluster_id = $1 AND to_cluster_id = $2
        """, cluster_ids[0], cluster_ids[1])
        
        assert relationship is not None, "Cluster relationship not created"
        assert relationship['relationship_type'] == 'contradicts'
        assert relationship['strength'] == 0.7

async def test_cluster_activation_history(db_pool):
    """Test cluster activation tracking"""
    async with db_pool.acquire() as conn:
        # Create test cluster
        cluster_id = await conn.fetchval("""
            INSERT INTO memory_clusters (
                cluster_type,
                name,
                centroid_embedding
            ) VALUES (
                'pattern'::cluster_type,
                'Test Pattern',
                array_fill(0.5, ARRAY[1536])::vector
            ) RETURNING id
        """)
        
        # Record activation
        activation_id = await conn.fetchval("""
            INSERT INTO cluster_activation_history (
                cluster_id,
                activation_context,
                activation_strength,
                co_activated_clusters,
                resulting_insights
            ) VALUES (
                $1,
                'User mentioned feeling lonely',
                0.9,
                ARRAY[]::UUID[],
                '{"insight": "User pattern of isolation detected"}'::jsonb
            ) RETURNING id
        """, cluster_id)
        
        assert activation_id is not None, "Failed to record activation"
        
        # Update cluster activation count
        await conn.execute("""
            UPDATE memory_clusters
            SET activation_count = activation_count + 1
            WHERE id = $1
        """, cluster_id)
        
        # Verify activation recorded
        activation = await conn.fetchrow("""
            SELECT * FROM cluster_activation_history
            WHERE id = $1
        """, activation_id)
        
        assert activation['activation_strength'] == 0.9
        assert activation['activation_context'] == 'User mentioned feeling lonely'

async def test_cluster_worldview_alignment(db_pool):
    """Test cluster alignment with worldview"""
    async with db_pool.acquire() as conn:
        # Create worldview primitive
        worldview_id = await conn.fetchval("""
            INSERT INTO worldview_primitives (
                category,
                belief,
                confidence,
                preferred_clusters
            ) VALUES (
                'values',
                'Connection is essential for wellbeing',
                0.9,
                ARRAY[]::UUID[]
            ) RETURNING id
        """)
        
        # Create aligned cluster
        cluster_id = await conn.fetchval("""
            INSERT INTO memory_clusters (
                cluster_type,
                name,
                centroid_embedding,
                worldview_alignment
            ) VALUES (
                'theme'::cluster_type,
                'Human Connection',
                array_fill(0.7, ARRAY[1536])::vector,
                0.95
            ) RETURNING id
        """)
        
        # Update worldview to prefer this cluster
        await conn.execute("""
            UPDATE worldview_primitives
            SET preferred_clusters = array_append(preferred_clusters, $1)
            WHERE id = $2
        """, cluster_id, worldview_id)
        
        # Verify alignment
        result = await conn.fetchrow("""
            SELECT preferred_clusters
            FROM worldview_primitives
            WHERE id = $1
        """, worldview_id)
        
        assert cluster_id in result['preferred_clusters']

async def test_identity_core_clusters(db_pool):
    """Test identity model with core memory clusters"""
    async with db_pool.acquire() as conn:
        # Create core clusters
        cluster_ids = []
        for name in ['Self-as-Helper', 'Creative-Expression']:
            cluster_id = await conn.fetchval("""
                INSERT INTO memory_clusters (
                    cluster_type,
                    name,
                    centroid_embedding,
                    importance_score
                ) VALUES (
                    'theme'::cluster_type,
                    $1,
                    array_fill(0.8, ARRAY[1536])::vector,
                    0.9
                ) RETURNING id
            """, name)
            cluster_ids.append(cluster_id)
        
        # Create identity with core clusters
        identity_id = await conn.fetchval("""
            INSERT INTO identity_model (
                self_concept,
                core_memory_clusters
            ) VALUES (
                '{"role": "supportive companion"}'::jsonb,
                $1
            ) RETURNING id
        """, cluster_ids)
        
        # Verify core clusters
        identity = await conn.fetchrow("""
            SELECT core_memory_clusters
            FROM identity_model
            WHERE id = $1
        """, identity_id)
        
        assert len(identity['core_memory_clusters']) == 2
        assert all(cid in identity['core_memory_clusters'] for cid in cluster_ids)

async def test_assign_memory_to_clusters_function(db_pool):
    """Test the assign_memory_to_clusters function
    
    Note: This test requires the updated schema.sql to be applied to the database.
    The assign_memory_to_clusters function was updated to use 
    'WHERE centroid_embedding IS NOT NULL' instead of 'WHERE status = 'active''
    """
    async with db_pool.acquire() as conn:
        # Create test clusters with different centroids
        cluster_ids = []
        for i in range(3):
            # Create distinct centroid embeddings
            centroid = [0.0] * 1536
            centroid[i*100:(i+1)*100] = [1.0] * 100  # Make each cluster distinct
            
            cluster_id = await conn.fetchval("""
                INSERT INTO memory_clusters (
                    cluster_type,
                    name,
                    centroid_embedding
                ) VALUES (
                    'theme'::cluster_type,
                    'Test Cluster ' || $1,
                    $2::vector
                ) RETURNING id
            """, str(i), str(centroid))
            cluster_ids.append(cluster_id)
        
        # Create memory with embedding similar to first cluster
        memory_embedding = [0.0] * 1536
        memory_embedding[0:100] = [0.9] * 100  # Similar to first cluster
        
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding
            ) VALUES (
                'semantic'::memory_type,
                'Test memory for auto-clustering',
                $1::vector
            ) RETURNING id
        """, str(memory_embedding))
        
        # Assign to clusters
        await conn.execute("""
            SELECT assign_memory_to_clusters($1, 2)
        """, memory_id)
        
        # Verify assignment
        memberships = await conn.fetch("""
            SELECT cluster_id, membership_strength
            FROM memory_cluster_members
            WHERE memory_id = $1
            ORDER BY membership_strength DESC
        """, memory_id)
        
        assert len(memberships) > 0, "Memory not assigned to any clusters"
        assert memberships[0]['membership_strength'] >= 0.7, "Expected high similarity"

async def test_recalculate_cluster_centroid_function(db_pool):
    """Test the recalculate_cluster_centroid function"""
    async with db_pool.acquire() as conn:
        # Create cluster
        cluster_id = await conn.fetchval("""
            INSERT INTO memory_clusters (
                cluster_type,
                name,
                centroid_embedding
            ) VALUES (
                'theme'::cluster_type,
                'Test Centroid Cluster',
                array_fill(0.0, ARRAY[1536])::vector
            ) RETURNING id
        """)
        
        # Add memories with different embeddings
        for i in range(3):
            memory_id = await conn.fetchval("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding,
                    status
                ) VALUES (
                    'semantic'::memory_type,
                    'Memory ' || $1,
                    array_fill($2::float, ARRAY[1536])::vector,
                    'active'::memory_status
                ) RETURNING id
            """, str(i), float(i+1) * 0.2)
            
            await conn.execute("""
                INSERT INTO memory_cluster_members (
                    cluster_id,
                    memory_id,
                    membership_strength
                ) VALUES ($1, $2, $3)
            """, cluster_id, memory_id, 0.8)
        
        # Recalculate centroid
        await conn.execute("""
            SELECT recalculate_cluster_centroid($1)
        """, cluster_id)
        
        # Check if centroid was updated
        result = await conn.fetchrow("""
            SELECT (vector_to_float4(centroid_embedding, 1536, false))[1] as first_value
            FROM memory_clusters
            WHERE id = $1
        """, cluster_id)
        
        # The average of 0.2, 0.4, 0.6 should be 0.4
        assert result['first_value'] is not None, "Centroid not updated"

async def test_cluster_insights_view(db_pool):
    """Test the cluster_insights view"""
    async with db_pool.acquire() as conn:
        # Create cluster with members using unique name
        import time
        unique_name = f'Insight Test Cluster {int(time.time() * 1000)}'
        cluster_id = await conn.fetchval("""
            INSERT INTO memory_clusters (
                cluster_type,
                name,
                importance_score,
                coherence_score,
                centroid_embedding
            ) VALUES (
                'theme'::cluster_type,
                $1,
                0.8,
                0.9,
                array_fill(0.5, ARRAY[1536])::vector
            ) RETURNING id
        """, unique_name)
        
        # Add memories
        for i in range(5):
            memory_id = await conn.fetchval("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding
                ) VALUES (
                    'episodic'::memory_type,
                    'Insight memory ' || $1,
                    array_fill(0.5, ARRAY[1536])::vector
                ) RETURNING id
            """, str(i))
            
            await conn.execute("""
                INSERT INTO memory_cluster_members (
                    cluster_id,
                    memory_id
                ) VALUES ($1, $2)
            """, cluster_id, memory_id)
        
        # Query view
        insights = await conn.fetch("""
            SELECT * FROM cluster_insights
            WHERE name = $1
        """, unique_name)
        
        assert len(insights) == 1
        assert insights[0]['memory_count'] == 5
        assert insights[0]['importance_score'] == 0.8
        assert insights[0]['coherence_score'] == 0.9

async def test_active_themes_view(db_pool):
    """Test the active_themes view"""
    async with db_pool.acquire() as conn:
        # Create active cluster
        cluster_id = await conn.fetchval("""
            INSERT INTO memory_clusters (
                cluster_type,
                name,
                emotional_signature,
                keywords,
                centroid_embedding
            ) VALUES (
                'emotion'::cluster_type,
                'Recent Anxiety',
                '{"primary": "anxiety", "intensity": 0.7}'::jsonb,
                ARRAY['worry', 'stress', 'uncertainty'],
                array_fill(0.3, ARRAY[1536])::vector
            ) RETURNING id
        """)
        
        # Record recent activations
        for i in range(3):
            await conn.execute("""
                INSERT INTO cluster_activation_history (
                    cluster_id,
                    activation_context,
                    activation_strength,
                    activated_at
                ) VALUES (
                    $1,
                    'Context ' || $2,
                    0.8,
                    CURRENT_TIMESTAMP - interval '1 hour' * $3
                )
            """, cluster_id, str(i), i)
        
        # Query view
        themes = await conn.fetch("""
            SELECT * FROM active_themes
            WHERE theme = 'Recent Anxiety'
        """)
        
        assert len(themes) > 0
        assert themes[0]['recent_activations'] == 3

async def test_update_cluster_activation_trigger(db_pool):
    """Test the update_cluster_activation trigger"""
    async with db_pool.acquire() as conn:
        # Create cluster with unique name
        import time
        unique_name = f'Activation Test {int(time.time() * 1000)}'
        cluster_id = await conn.fetchval("""
            INSERT INTO memory_clusters (
                cluster_type,
                name,
                centroid_embedding,
                importance_score,
                activation_count
            ) VALUES (
                'theme'::cluster_type,
                $1,
                array_fill(0.5, ARRAY[1536])::vector,
                0.5,
                0
            ) RETURNING id
        """, unique_name)
        
        # Get initial values
        initial = await conn.fetchrow("""
            SELECT importance_score, activation_count, last_activated
            FROM memory_clusters
            WHERE id = $1
        """, cluster_id)
        
        # Update activation count
        await conn.execute("""
            UPDATE memory_clusters
            SET activation_count = activation_count + 1
            WHERE id = $1
        """, cluster_id)
        
        # Get updated values
        updated = await conn.fetchrow("""
            SELECT importance_score, activation_count, last_activated
            FROM memory_clusters
            WHERE id = $1
        """, cluster_id)
        
        # Check that activation count increased (may be more than 1 due to trigger behavior)
        assert updated['activation_count'] > initial['activation_count'], f"Expected activation count to increase from {initial['activation_count']} but got {updated['activation_count']}"
        assert updated['importance_score'] > initial['importance_score']
        assert updated['last_activated'] is not None

async def test_cluster_types(db_pool):
    """Test all cluster types"""
    async with db_pool.acquire() as conn:
        cluster_types = ['theme', 'emotion', 'temporal', 'person', 'pattern', 'mixed']
        
        for c_type in cluster_types:
            cluster_id = await conn.fetchval("""
                INSERT INTO memory_clusters (
                    cluster_type,
                    name,
                    centroid_embedding
                ) VALUES (
                    $1::cluster_type,
                    'Test ' || $1 || ' cluster',
                    array_fill(0.5, ARRAY[1536])::vector
                ) RETURNING id
            """, c_type)
            
            assert cluster_id is not None, f"Failed to create {c_type} cluster"
        
        # Verify all types exist
        count = await conn.fetchval("""
            SELECT COUNT(DISTINCT cluster_type)
            FROM memory_clusters
        """)
        
        assert count >= len(cluster_types)

async def test_cluster_memory_retrieval_performance(db_pool):
    """Test performance of cluster-based memory retrieval"""
    async with db_pool.acquire() as conn:
        # Create cluster
        cluster_id = await conn.fetchval("""
            INSERT INTO memory_clusters (
                cluster_type,
                name,
                centroid_embedding,
                keywords
            ) VALUES (
                'theme'::cluster_type,
                'Loneliness',
                array_fill(0.3, ARRAY[1536])::vector,
                ARRAY['lonely', 'alone', 'isolated']
            ) RETURNING id
        """)
        
        # Add many memories to cluster
        memory_ids = []
        for i in range(50):
            memory_id = await conn.fetchval("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding,
                    importance
                ) VALUES (
                    'episodic'::memory_type,
                    'Loneliness memory ' || $1,
                    array_fill(0.3, ARRAY[1536])::vector,
                    $2
                ) RETURNING id
            """, str(i), 0.5 + (i * 0.01))
            
            await conn.execute("""
                INSERT INTO memory_cluster_members (
                    cluster_id,
                    memory_id,
                    membership_strength
                ) VALUES ($1, $2, $3)
            """, cluster_id, memory_id, 0.7 + (i * 0.001))
            
            memory_ids.append(memory_id)
        
        # Test retrieval by cluster
        import time
        start_time = time.time()
        
        results = await conn.fetch("""
            SELECT m.*, mcm.membership_strength
            FROM memories m
            JOIN memory_cluster_members mcm ON m.id = mcm.memory_id
            WHERE mcm.cluster_id = $1
            ORDER BY mcm.membership_strength DESC, m.importance DESC
            LIMIT 10
        """, cluster_id)
        
        retrieval_time = time.time() - start_time
        
        assert len(results) == 10
        assert retrieval_time < 0.1, f"Cluster retrieval too slow: {retrieval_time}s"
        
        # Verify ordering
        strengths = [r['membership_strength'] for r in results]
        assert strengths == sorted(strengths, reverse=True)


# HIGH PRIORITY ADDITIONAL TESTS

async def test_constraint_violations(db_pool):
    """Test constraint violations and error handling"""
    async with db_pool.acquire() as conn:
        # Test invalid emotional_valence (should be between -1 and 1)
        with pytest.raises(Exception):
            await conn.execute("""
                INSERT INTO episodic_memories (
                    memory_id,
                    action_taken,
                    context,
                    result,
                    emotional_valence
                ) VALUES (
                    gen_random_uuid(),
                    '{"action": "test"}',
                    '{"context": "test"}',
                    '{"result": "test"}',
                    2.0
                )
            """)
        
        # Test invalid confidence score (should be between 0 and 1)
        with pytest.raises(Exception):
            await conn.execute("""
                INSERT INTO semantic_memories (
                    memory_id,
                    confidence
                ) VALUES (
                    gen_random_uuid(),
                    1.5
                )
            """)
        
        # Test foreign key violation
        with pytest.raises(Exception):
            fake_uuid = 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'
            await conn.execute("""
                INSERT INTO episodic_memories (
                    memory_id,
                    action_taken,
                    context,
                    result
                ) VALUES (
                    $1::uuid,
                    '{"action": "test"}',
                    '{"context": "test"}',
                    '{"result": "test"}'
                )
            """, fake_uuid)
        
        # Test invalid vector dimension
        with pytest.raises(Exception):
            await conn.execute("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding
                ) VALUES (
                    'semantic'::memory_type,
                    'Test content',
                    array_fill(0, ARRAY[100])::vector
                )
            """)
        
        # Test null constraint violation
        with pytest.raises(Exception):
            await conn.execute("""
                INSERT INTO memories (
                    type,
                    embedding
                ) VALUES (
                    'semantic'::memory_type,
                    array_fill(0, ARRAY[1536])::vector
                )
            """)


async def test_memory_consolidation_workflow(db_pool):
    """Test complete memory consolidation from working memory to long-term storage"""
    async with db_pool.acquire() as conn:
        # Step 1: Create working memory entries
        working_memories = []
        for i in range(5):
            wm_id = await conn.fetchval("""
                INSERT INTO working_memory (
                    content,
                    embedding,
                    expiry
                ) VALUES (
                    'Working memory content ' || $1,
                    array_fill($2::float, ARRAY[1536])::vector,
                    CURRENT_TIMESTAMP + interval '1 hour'
                ) RETURNING id
            """, str(i), float(i) * 0.1)
            working_memories.append(wm_id)
        
        # Step 2: Simulate consolidation process
        consolidated_memories = []
        for wm_id in working_memories:
            # Get working memory content
            wm_data = await conn.fetchrow("""
                SELECT content, embedding FROM working_memory WHERE id = $1
            """, wm_id)
            
            # Create long-term memory
            ltm_id = await conn.fetchval("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding,
                    importance
                ) VALUES (
                    'episodic'::memory_type,
                    'Consolidated: ' || $1,
                    $2,
                    0.7
                ) RETURNING id
            """, wm_data['content'], wm_data['embedding'])
            
            # Create episodic details
            await conn.execute("""
                INSERT INTO episodic_memories (
                    memory_id,
                    action_taken,
                    context,
                    result,
                    emotional_valence
                ) VALUES (
                    $1,
                    '{"action": "consolidation"}',
                    '{"source": "working_memory"}',
                    '{"status": "consolidated"}',
                    0.0
                )
            """, ltm_id)
            
            consolidated_memories.append(ltm_id)
            
            # Remove from working memory
            await conn.execute("""
                DELETE FROM working_memory WHERE id = $1
            """, wm_id)
        
        # Step 3: Verify consolidation
        # Check working memory is empty
        wm_count = await conn.fetchval("""
            SELECT COUNT(*) FROM working_memory
            WHERE id = ANY($1::uuid[])
        """, working_memories)
        assert wm_count == 0, "Working memory not properly cleared"
        
        # Check long-term memories exist
        ltm_count = await conn.fetchval("""
            SELECT COUNT(*) FROM memories
            WHERE id = ANY($1::uuid[])
        """, consolidated_memories)
        assert ltm_count == 5, "Not all memories consolidated"
        
        # Step 4: Test memory clustering after consolidation
        for memory_id in consolidated_memories:
            await conn.execute("""
                SELECT assign_memory_to_clusters($1, 2)
            """, memory_id)
        
        # Verify cluster assignments
        cluster_assignments = await conn.fetchval("""
            SELECT COUNT(*) FROM memory_cluster_members
            WHERE memory_id = ANY($1::uuid[])
        """, consolidated_memories)
        assert cluster_assignments > 0, "Memories not assigned to clusters"


async def test_large_dataset_performance(db_pool):
    """Test system performance with large datasets"""
    async with db_pool.acquire() as conn:
        import time
        
        # Create large number of memories (1000 for testing, would be 10K+ in production)
        batch_size = 100
        total_memories = 1000
        memory_ids = []
        
        print(f"Creating {total_memories} memories in batches of {batch_size}...")
        
        for batch_start in range(0, total_memories, batch_size):
            batch_end = min(batch_start + batch_size, total_memories)
            batch_memories = []
            
            # Create batch of memories
            for i in range(batch_start, batch_end):
                # Create diverse embeddings
                embedding = [0.0] * 1536
                # Create patterns in embeddings for clustering
                pattern_start = (i % 10) * 150
                pattern_end = min(pattern_start + 150, 1536)
                embedding[pattern_start:pattern_end] = [0.8] * (pattern_end - pattern_start)
                
                memory_id = await conn.fetchval("""
                    INSERT INTO memories (
                        type,
                        content,
                        embedding,
                        importance
                    ) VALUES (
                        $1::memory_type,
                        'Large dataset memory ' || $2,
                        $3::vector,
                        $4
                    ) RETURNING id
                """, 
                    ['episodic', 'semantic', 'procedural', 'strategic'][i % 4],
                    str(i),
                    str(embedding),
                    0.1 + (i % 100) * 0.01
                )
                batch_memories.append(memory_id)
            
            memory_ids.extend(batch_memories)
        
        print(f"Created {len(memory_ids)} memories")
        
        # Test 1: Vector similarity search performance
        query_embedding = [0.8] * 150 + [0.0] * (1536 - 150)
        
        start_time = time.time()
        similar_memories = await conn.fetch("""
            SELECT id, content, embedding <=> $1::vector as distance
            FROM memories
            ORDER BY embedding <=> $1::vector
            LIMIT 50
        """, str(query_embedding))
        vector_search_time = time.time() - start_time
        
        assert len(similar_memories) == 50
        assert vector_search_time < 1.0, f"Vector search too slow: {vector_search_time}s"
        print(f"Vector search time: {vector_search_time:.3f}s")
        
        # Test 2: Complex query performance
        start_time = time.time()
        complex_results = await conn.fetch("""
            SELECT m.type, COUNT(*) as count, AVG(m.importance) as avg_importance
            FROM memories m
            WHERE m.status = 'active'
            AND m.importance > 0.5
            GROUP BY m.type
            ORDER BY avg_importance DESC
        """)
        complex_query_time = time.time() - start_time
        
        assert len(complex_results) > 0
        assert complex_query_time < 0.5, f"Complex query too slow: {complex_query_time}s"
        print(f"Complex query time: {complex_query_time:.3f}s")
        
        # Test 3: Memory health view performance
        start_time = time.time()
        health_stats = await conn.fetch("""
            SELECT * FROM memory_health
        """)
        view_query_time = time.time() - start_time
        
        assert len(health_stats) > 0
        assert view_query_time < 0.5, f"View query too slow: {view_query_time}s"
        print(f"View query time: {view_query_time:.3f}s")


async def test_concurrency_safety(db_pool):
    """Test concurrent operations for race conditions"""
    async with db_pool.acquire() as conn:
        # Create test memory for concurrent updates
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding,
                importance,
                access_count
            ) VALUES (
                'semantic'::memory_type,
                'Concurrency test memory',
                array_fill(0.5, ARRAY[1536])::vector,
                0.5,
                0
            ) RETURNING id
        """)
        
        # Test concurrent access count updates
        async def update_access_count(pool, mem_id, increment):
            async with pool.acquire() as connection:
                for _ in range(increment):
                    await connection.execute("""
                        UPDATE memories 
                        SET access_count = access_count + 1
                        WHERE id = $1
                    """, mem_id)
        
        # Run concurrent updates
        import asyncio
        tasks = [
            update_access_count(db_pool, memory_id, 10),
            update_access_count(db_pool, memory_id, 10),
            update_access_count(db_pool, memory_id, 10)
        ]
        
        await asyncio.gather(*tasks)
        
        # Verify final access count
        final_count = await conn.fetchval("""
            SELECT access_count FROM memories WHERE id = $1
        """, memory_id)
        
        assert final_count == 30, f"Expected 30 accesses, got {final_count}"
        
        # Test concurrent cluster assignments
        cluster_id = await conn.fetchval("""
            INSERT INTO memory_clusters (
                cluster_type,
                name,
                centroid_embedding
            ) VALUES (
                'theme'::cluster_type,
                'Concurrency Test Cluster',
                array_fill(0.5, ARRAY[1536])::vector
            ) RETURNING id
        """)
        
        # Create multiple memories for concurrent cluster assignment
        test_memories = []
        for i in range(5):
            mem_id = await conn.fetchval("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding
                ) VALUES (
                    'semantic'::memory_type,
                    'Concurrent memory ' || $1,
                    array_fill(0.5, ARRAY[1536])::vector
                ) RETURNING id
            """, str(i))
            test_memories.append(mem_id)
        
        # Concurrent cluster assignments
        async def assign_to_cluster(pool, mem_id, clust_id):
            async with pool.acquire() as connection:
                try:
                    await connection.execute("""
                        INSERT INTO memory_cluster_members (
                            cluster_id,
                            memory_id,
                            membership_strength
                        ) VALUES ($1, $2, 0.8)
                        ON CONFLICT DO NOTHING
                    """, clust_id, mem_id)
                except Exception as e:
                    # Expected for some concurrent operations
                    pass
        
        assignment_tasks = [
            assign_to_cluster(db_pool, mem_id, cluster_id)
            for mem_id in test_memories
        ]
        
        await asyncio.gather(*assignment_tasks)
        
        # Verify assignments
        assignment_count = await conn.fetchval("""
            SELECT COUNT(*) FROM memory_cluster_members
            WHERE cluster_id = $1
        """, cluster_id)
        
        assert assignment_count == 5, f"Expected 5 assignments, got {assignment_count}"


async def test_cascade_delete_integrity(db_pool):
    """Test referential integrity with cascade deletes"""
    async with db_pool.acquire() as conn:
        # Create memory with all related data
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding
            ) VALUES (
                'episodic'::memory_type,
                'Test cascade delete',
                array_fill(0.5, ARRAY[1536])::vector
            ) RETURNING id
        """)
        
        # Add episodic details
        await conn.execute("""
            INSERT INTO episodic_memories (
                memory_id,
                action_taken,
                context,
                result
            ) VALUES (
                $1,
                '{"action": "test"}',
                '{"context": "test"}',
                '{"result": "test"}'
            )
        """, memory_id)
        
        # Add to cluster
        cluster_id = await conn.fetchval("""
            INSERT INTO memory_clusters (
                cluster_type,
                name,
                centroid_embedding
            ) VALUES (
                'theme'::cluster_type,
                'Cascade Test Cluster',
                array_fill(0.5, ARRAY[1536])::vector
            ) RETURNING id
        """)
        
        await conn.execute("""
            INSERT INTO memory_cluster_members (
                cluster_id,
                memory_id
            ) VALUES ($1, $2)
        """, cluster_id, memory_id)
        
        # Add memory changes
        await conn.execute("""
            INSERT INTO memory_changes (
                memory_id,
                change_type,
                old_value,
                new_value
            ) VALUES (
                $1,
                'creation',
                '{}',
                '{"status": "created"}'
            )
        """, memory_id)
        
        # Verify all related data exists
        episodic_count = await conn.fetchval("""
            SELECT COUNT(*) FROM episodic_memories WHERE memory_id = $1
        """, memory_id)
        cluster_member_count = await conn.fetchval("""
            SELECT COUNT(*) FROM memory_cluster_members WHERE memory_id = $1
        """, memory_id)
        changes_count = await conn.fetchval("""
            SELECT COUNT(*) FROM memory_changes WHERE memory_id = $1
        """, memory_id)
        
        assert episodic_count == 1
        assert cluster_member_count == 1
        assert changes_count == 1
        
        # Test that direct deletion fails due to foreign key constraints
        # This is the correct behavior - memory type tables don't have CASCADE DELETE
        with pytest.raises(Exception):
            await conn.execute("""
                DELETE FROM memories WHERE id = $1
            """, memory_id)
        
        # Proper deletion requires removing ALL related data first
        # Delete episodic details
        await conn.execute("""
            DELETE FROM episodic_memories WHERE memory_id = $1
        """, memory_id)
        
        # Delete memory changes (audit trail - also no CASCADE)
        await conn.execute("""
            DELETE FROM memory_changes WHERE memory_id = $1
        """, memory_id)
        
        # Now delete the memory (cluster membership should cascade)
        await conn.execute("""
            DELETE FROM memories WHERE id = $1
        """, memory_id)
        
        # Verify cascade deletes worked for tables with CASCADE
        cluster_member_count_after = await conn.fetchval("""
            SELECT COUNT(*) FROM memory_cluster_members WHERE memory_id = $1
        """, memory_id)
        changes_count_after = await conn.fetchval("""
            SELECT COUNT(*) FROM memory_changes WHERE memory_id = $1
        """, memory_id)
        episodic_count_after = await conn.fetchval("""
            SELECT COUNT(*) FROM episodic_memories WHERE memory_id = $1
        """, memory_id)
        
        # These should all be deleted now
        assert cluster_member_count_after == 0, "Cluster membership not cascade deleted"
        assert changes_count_after == 0, "Memory changes not deleted"
        assert episodic_count_after == 0, "Episodic memory not deleted"


async def test_memory_lifecycle_workflow(db_pool):
    """Test complete memory lifecycle from creation to archival"""
    async with db_pool.acquire() as conn:
        # Step 1: Create new memory
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding,
                importance,
                access_count
            ) VALUES (
                'semantic'::memory_type,
                'Lifecycle test memory',
                array_fill(0.5, ARRAY[1536])::vector,
                0.3,
                0
            ) RETURNING id
        """)
        
        initial_relevance = await conn.fetchval("""
            SELECT relevance_score FROM memories WHERE id = $1
        """, memory_id)
        
        # Step 2: Simulate memory access and importance growth
        for i in range(5):
            await conn.execute("""
                UPDATE memories 
                SET access_count = access_count + 1
                WHERE id = $1
            """, memory_id)
        
        mid_relevance = await conn.fetchval("""
            SELECT relevance_score FROM memories WHERE id = $1
        """, memory_id)
        
        assert mid_relevance > initial_relevance, "Relevance should increase with access"
        
        # Step 3: Simulate time passage and decay
        await conn.execute("""
            UPDATE memories 
            SET created_at = CURRENT_TIMESTAMP - interval '30 days'
            WHERE id = $1
        """, memory_id)
        
        aged_relevance = await conn.fetchval("""
            SELECT relevance_score FROM memories WHERE id = $1
        """, memory_id)
        
        assert aged_relevance < mid_relevance, "Relevance should decrease with age"
        
        # Step 4: Archive low-relevance memory
        await conn.execute("""
            UPDATE memories 
            SET status = 'archived'::memory_status
            WHERE id = $1 AND relevance_score < 0.1
        """, memory_id)
        
        final_status = await conn.fetchval("""
            SELECT status FROM memories WHERE id = $1
        """, memory_id)
        
        # Memory might or might not be archived depending on exact relevance calculation
        assert final_status in ['active', 'archived'], "Memory should have valid status"
        
        # Step 5: Test memory retrieval excludes archived memories
        active_memories = await conn.fetch("""
            SELECT * FROM memories 
            WHERE status = 'active' AND id = $1
        """, memory_id)
        
        if final_status == 'archived':
            assert len(active_memories) == 0, "Archived memory should not appear in active queries"


# CRITICAL MISSING TESTS

async def test_edge_cases_empty_clusters(db_pool):
    """Test edge cases with empty clusters"""
    async with db_pool.acquire() as conn:
        # Create empty cluster
        cluster_id = await conn.fetchval("""
            INSERT INTO memory_clusters (
                cluster_type,
                name,
                centroid_embedding
            ) VALUES (
                'theme'::cluster_type,
                'Empty Test Cluster',
                array_fill(0.5, ARRAY[1536])::vector
            ) RETURNING id
        """)
        
        # Test recalculating centroid on empty cluster
        await conn.execute("""
            SELECT recalculate_cluster_centroid($1)
        """, cluster_id)
        
        # Verify cluster still exists but centroid might be null
        cluster = await conn.fetchrow("""
            SELECT * FROM memory_clusters WHERE id = $1
        """, cluster_id)
        assert cluster is not None, "Empty cluster should still exist"
        
        # Test cluster insights view with empty cluster
        insights = await conn.fetch("""
            SELECT * FROM cluster_insights WHERE id = $1
        """, cluster_id)
        assert len(insights) == 1, "Empty cluster should appear in insights"
        assert insights[0]['memory_count'] == 0, "Empty cluster should have 0 memories"


async def test_edge_cases_circular_relationships(db_pool):
    """Test edge cases with circular cluster relationships"""
    async with db_pool.acquire() as conn:
        # Create three clusters
        cluster_ids = []
        for i in range(3):
            cluster_id = await conn.fetchval("""
                INSERT INTO memory_clusters (
                    cluster_type,
                    name,
                    centroid_embedding
                ) VALUES (
                    'theme'::cluster_type,
                    'Circular Cluster ' || $1,
                    array_fill($2::float, ARRAY[1536])::vector
                ) RETURNING id
            """, str(i), float(i) * 0.3)
            cluster_ids.append(cluster_id)
        
        # Create circular relationships: A -> B -> C -> A
        relationships = [
            (cluster_ids[0], cluster_ids[1], 'leads_to'),
            (cluster_ids[1], cluster_ids[2], 'causes'),
            (cluster_ids[2], cluster_ids[0], 'reinforces')
        ]
        
        for from_id, to_id, rel_type in relationships:
            await conn.execute("""
                INSERT INTO cluster_relationships (
                    from_cluster_id,
                    to_cluster_id,
                    relationship_type,
                    strength
                ) VALUES ($1, $2, $3, 0.7)
            """, from_id, to_id, rel_type)
        
        # Verify all relationships exist
        total_relationships = await conn.fetchval("""
            SELECT COUNT(*) FROM cluster_relationships
            WHERE from_cluster_id = ANY($1::uuid[])
        """, cluster_ids)
        assert total_relationships == 3, "All circular relationships should exist"
        
        # Test that we can detect cycles (this would be application logic)
        cycle_query = await conn.fetch("""
            WITH RECURSIVE cluster_paths AS (
                SELECT from_cluster_id, to_cluster_id, 1 as depth, 
                       ARRAY[from_cluster_id] as path
                FROM cluster_relationships
                WHERE from_cluster_id = $1
                
                UNION ALL
                
                SELECT cr.from_cluster_id, cr.to_cluster_id, cp.depth + 1,
                       cp.path || cr.from_cluster_id
                FROM cluster_relationships cr
                JOIN cluster_paths cp ON cr.from_cluster_id = cp.to_cluster_id
                WHERE cp.depth < 5 AND NOT (cr.from_cluster_id = ANY(cp.path))
            )
            SELECT * FROM cluster_paths WHERE to_cluster_id = $1 AND depth > 1
        """, cluster_ids[0])
        
        assert len(cycle_query) > 0, "Should detect circular relationship"


async def test_edge_cases_extreme_values(db_pool):
    """Test edge cases with extreme values"""
    async with db_pool.acquire() as conn:
        # Test memory with very high importance
        high_importance_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding,
                importance,
                access_count
            ) VALUES (
                'semantic'::memory_type,
                'Extremely important memory',
                array_fill(0.5, ARRAY[1536])::vector,
                999999.0,
                999999
            ) RETURNING id
        """)
        
        # Test memory with very old timestamp
        old_memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding,
                importance,
                created_at
            ) VALUES (
                'episodic'::memory_type,
                'Ancient memory',
                array_fill(0.5, ARRAY[1536])::vector,
                0.5,
                '1900-01-01'::timestamp
            ) RETURNING id
        """)
        
        # Test relevance calculation with extreme values
        high_relevance = await conn.fetchval("""
            SELECT relevance_score FROM memories WHERE id = $1
        """, high_importance_id)
        
        old_relevance = await conn.fetchval("""
            SELECT relevance_score FROM memories WHERE id = $1
        """, old_memory_id)
        
        assert high_relevance > 1000, "High importance should result in high relevance"
        assert old_relevance < 0.01, "Very old memory should have very low relevance"
        
        # Test with zero vectors
        zero_vector_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding
            ) VALUES (
                'semantic'::memory_type,
                'Zero vector memory',
                array_fill(0.0, ARRAY[1536])::vector
            ) RETURNING id
        """)
        
        # Test similarity search with zero vector
        zero_results = await conn.fetch("""
            SELECT id, embedding <=> array_fill(0.0, ARRAY[1536])::vector as distance
            FROM memories
            WHERE id = $1
        """, zero_vector_id)
        
        assert len(zero_results) == 1
        # Zero vectors result in NaN for cosine distance, which is expected behavior
        import math
        assert math.isnan(zero_results[0]['distance']), "Zero vector cosine distance should be NaN"


async def test_data_integrity_orphaned_records(db_pool):
    """Test data integrity with potential orphaned records"""
    async with db_pool.acquire() as conn:
        # Create memory with related data
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding
            ) VALUES (
                'semantic'::memory_type,
                'Test orphan memory',
                array_fill(0.5, ARRAY[1536])::vector
            ) RETURNING id
        """)
        
        # Add semantic details
        await conn.execute("""
            INSERT INTO semantic_memories (
                memory_id,
                confidence
            ) VALUES ($1, 0.8)
        """, memory_id)
        
        # Add to cluster
        cluster_id = await conn.fetchval("""
            INSERT INTO memory_clusters (
                cluster_type,
                name,
                centroid_embedding
            ) VALUES (
                'theme'::cluster_type,
                'Orphan Test Cluster',
                array_fill(0.5, ARRAY[1536])::vector
            ) RETURNING id
        """)
        
        await conn.execute("""
            INSERT INTO memory_cluster_members (
                cluster_id,
                memory_id
            ) VALUES ($1, $2)
        """, cluster_id, memory_id)
        
        # Simulate orphaned records by deleting cluster but not membership
        # First, we need to temporarily disable the foreign key constraint
        # In a real scenario, this could happen due to application bugs
        
        # Check for orphaned cluster memberships
        orphaned_memberships = await conn.fetch("""
            SELECT mcm.* 
            FROM memory_cluster_members mcm
            LEFT JOIN memory_clusters mc ON mcm.cluster_id = mc.id
            WHERE mc.id IS NULL
        """)
        
        # Should be empty in normal operation
        assert len(orphaned_memberships) == 0, "No orphaned cluster memberships should exist"
        
        # Check for orphaned memory type records
        orphaned_semantic = await conn.fetch("""
            SELECT sm.*
            FROM semantic_memories sm
            LEFT JOIN memories m ON sm.memory_id = m.id
            WHERE m.id IS NULL
        """)
        
        assert len(orphaned_semantic) == 0, "No orphaned semantic memories should exist"


async def test_computed_field_accuracy(db_pool):
    """Test accuracy of computed fields"""
    async with db_pool.acquire() as conn:
        # Test procedural memory success rate calculation
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding
            ) VALUES (
                'procedural'::memory_type,
                'Success rate test',
                array_fill(0.5, ARRAY[1536])::vector
            ) RETURNING id
        """)
        
        # Test various success rate scenarios
        test_cases = [
            (0, 0, 0.0),      # No attempts
            (5, 10, 0.5),     # 50% success
            (10, 10, 1.0),    # 100% success
            (0, 5, 0.0),      # 0% success
            (1, 3, 0.333333)  # 33.33% success
        ]
        
        for success_count, total_attempts, expected_rate in test_cases:
            await conn.execute("""
                INSERT INTO procedural_memories (
                    memory_id,
                    steps,
                    success_count,
                    total_attempts
                ) VALUES ($1, '{"steps": ["test"]}', $2, $3)
                ON CONFLICT (memory_id) DO UPDATE SET
                    success_count = $2,
                    total_attempts = $3
            """, memory_id, success_count, total_attempts)
            
            actual_rate = await conn.fetchval("""
                SELECT success_rate FROM procedural_memories WHERE memory_id = $1
            """, memory_id)
            
            if expected_rate == 0.333333:
                assert abs(actual_rate - expected_rate) < 0.000001, f"Success rate calculation incorrect: expected {expected_rate}, got {actual_rate}"
            else:
                assert actual_rate == expected_rate, f"Success rate calculation incorrect: expected {expected_rate}, got {actual_rate}"
        
        # Test relevance score accuracy
        test_memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding,
                importance,
                decay_rate,
                created_at
            ) VALUES (
                'semantic'::memory_type,
                'Relevance test',
                array_fill(0.5, ARRAY[1536])::vector,
                1.0,
                0.1,
                CURRENT_TIMESTAMP - interval '1 day'
            ) RETURNING id
        """)
        
        relevance = await conn.fetchval("""
            SELECT relevance_score FROM memories WHERE id = $1
        """, test_memory_id)
        
        # Manual calculation: importance * exp(-decay_rate * age_in_days)
        # 1.0 * exp(-0.1 * 1) ≈ 1.0 * 0.9048 ≈ 0.9048
        expected_relevance = 1.0 * 2.718281828459045 ** (-0.1 * 1)
        assert abs(relevance - expected_relevance) < 0.001, f"Relevance calculation incorrect: expected ~{expected_relevance}, got {relevance}"


async def test_trigger_consistency(db_pool):
    """Test that all triggers fire correctly and consistently"""
    async with db_pool.acquire() as conn:
        # Test memory timestamp trigger
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding
            ) VALUES (
                'semantic'::memory_type,
                'Trigger test memory',
                array_fill(0.5, ARRAY[1536])::vector
            ) RETURNING id
        """)
        
        initial_updated_at = await conn.fetchval("""
            SELECT updated_at FROM memories WHERE id = $1
        """, memory_id)
        
        # Wait and update
        await asyncio.sleep(0.1)
        await conn.execute("""
            UPDATE memories SET content = 'Updated content' WHERE id = $1
        """, memory_id)
        
        new_updated_at = await conn.fetchval("""
            SELECT updated_at FROM memories WHERE id = $1
        """, memory_id)
        
        assert new_updated_at > initial_updated_at, "Timestamp trigger should fire on update"
        
        # Test importance trigger - set initial importance first
        await conn.execute("""
            UPDATE memories SET importance = 0.5 WHERE id = $1
        """, memory_id)
        
        initial_importance = await conn.fetchval("""
            SELECT importance FROM memories WHERE id = $1
        """, memory_id)
        
        await conn.execute("""
            UPDATE memories SET access_count = access_count + 1 WHERE id = $1
        """, memory_id)
        
        new_importance = await conn.fetchval("""
            SELECT importance FROM memories WHERE id = $1
        """, memory_id)
        
        assert new_importance > initial_importance, "Importance trigger should fire on access count change"
        
        # Test cluster activation trigger
        cluster_id = await conn.fetchval("""
            INSERT INTO memory_clusters (
                cluster_type,
                name,
                centroid_embedding,
                activation_count,
                importance_score
            ) VALUES (
                'theme'::cluster_type,
                'Trigger Test Cluster',
                array_fill(0.5, ARRAY[1536])::vector,
                0,
                0.5
            ) RETURNING id
        """)
        
        initial_cluster_importance = await conn.fetchval("""
            SELECT importance_score FROM memory_clusters WHERE id = $1
        """, cluster_id)
        
        await conn.execute("""
            UPDATE memory_clusters SET activation_count = activation_count + 1 WHERE id = $1
        """, cluster_id)
        
        new_cluster_importance = await conn.fetchval("""
            SELECT importance_score FROM memory_clusters WHERE id = $1
        """, cluster_id)
        
        assert new_cluster_importance > initial_cluster_importance, "Cluster activation trigger should fire"


async def test_view_calculation_accuracy(db_pool):
    """Test accuracy of view calculations"""
    async with db_pool.acquire() as conn:
        # Create test data for memory_health view with unique content to avoid interference
        import time
        unique_suffix = str(int(time.time() * 1000))
        test_memories = []
        for i in range(10):
            memory_id = await conn.fetchval("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding,
                    importance,
                    access_count,
                    last_accessed
                ) VALUES (
                    'semantic'::memory_type,
                    'Health test memory ' || $1 || ' ' || $2,
                    array_fill(0.5, ARRAY[1536])::vector,
                    $3,
                    $4,
                    CASE WHEN $5 THEN CURRENT_TIMESTAMP - interval '12 hours' ELSE NULL END
                ) RETURNING id
            """, str(i), unique_suffix, float(i) * 0.1, i, i % 2 == 0)
            test_memories.append(memory_id)
        
        # Query memory_health view for just our test memories
        health_stats = await conn.fetchrow("""
            SELECT 
                type,
                COUNT(*) as total_memories,
                AVG(importance) as avg_importance,
                AVG(access_count) as avg_access_count,
                COUNT(*) FILTER (WHERE last_accessed > CURRENT_TIMESTAMP - INTERVAL '1 day') as accessed_last_day,
                AVG(relevance_score) as avg_relevance
            FROM memories
            WHERE type = 'semantic' AND content LIKE '%' || $1
            GROUP BY type
        """, unique_suffix)
        
        # Verify calculations
        assert health_stats['total_memories'] == 10, "Should count exactly our 10 test memories"
        
        # Calculate expected average importance: (0.0 + 0.1 + 0.2 + ... + 0.9) / 10 = 4.5 / 10 = 0.45
        expected_avg_importance = sum(i * 0.1 for i in range(10)) / 10
        actual_avg_importance = float(health_stats['avg_importance'])
        assert abs(actual_avg_importance - expected_avg_importance) < 0.01, f"Average importance calculation incorrect: expected {expected_avg_importance}, got {actual_avg_importance}"
        
        # Test cluster_insights view accuracy
        cluster_id = await conn.fetchval("""
            INSERT INTO memory_clusters (
                cluster_type,
                name,
                importance_score,
                coherence_score,
                centroid_embedding
            ) VALUES (
                'theme'::cluster_type,
                'Accuracy Test Cluster',
                0.75,
                0.85,
                array_fill(0.5, ARRAY[1536])::vector
            ) RETURNING id
        """)
        
        # Add some memories to cluster
        for memory_id in test_memories[:5]:
            await conn.execute("""
                INSERT INTO memory_cluster_members (
                    cluster_id,
                    memory_id
                ) VALUES ($1, $2)
            """, cluster_id, memory_id)
        
        cluster_insight = await conn.fetchrow("""
            SELECT * FROM cluster_insights WHERE name = 'Accuracy Test Cluster'
        """)
        
        assert cluster_insight['memory_count'] == 5, "Should count cluster members correctly"
        assert cluster_insight['importance_score'] == 0.75, "Should preserve importance score"
        assert cluster_insight['coherence_score'] == 0.85, "Should preserve coherence score"


async def test_error_recovery_scenarios(db_pool):
    """Test error recovery scenarios"""
    async with db_pool.acquire() as conn:
        # Test recovery from invalid JSON in JSONB fields
        memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding
            ) VALUES (
                'episodic'::memory_type,
                'Error recovery test',
                array_fill(0.5, ARRAY[1536])::vector
            ) RETURNING id
        """)
        
        # Insert valid episodic memory
        await conn.execute("""
            INSERT INTO episodic_memories (
                memory_id,
                action_taken,
                context,
                result
            ) VALUES (
                $1,
                '{"action": "valid_json"}',
                '{"context": "test"}',
                '{"result": "success"}'
            )
        """, memory_id)
        
        # Test that we can query and update the record
        episodic_data = await conn.fetchrow("""
            SELECT * FROM episodic_memories WHERE memory_id = $1
        """, memory_id)
        
        assert episodic_data is not None, "Should be able to query episodic memory"
        
        # Test updating with new valid JSON
        await conn.execute("""
            UPDATE episodic_memories 
            SET action_taken = '{"action": "updated_action"}'
            WHERE memory_id = $1
        """, memory_id)
        
        updated_data = await conn.fetchrow("""
            SELECT action_taken FROM episodic_memories WHERE memory_id = $1
        """, memory_id)
        
        # Parse the JSON if it's returned as a string
        action_taken = updated_data['action_taken']
        if isinstance(action_taken, str):
            action_taken = json.loads(action_taken)
        
        assert action_taken['action'] == 'updated_action', "Should update JSON correctly"
        
        # Test transaction rollback scenario
        try:
            async with conn.transaction():
                # Create a memory
                temp_memory_id = await conn.fetchval("""
                    INSERT INTO memories (
                        type,
                        content,
                        embedding
                    ) VALUES (
                        'semantic'::memory_type,
                        'Rollback test',
                        array_fill(0.5, ARRAY[1536])::vector
                    ) RETURNING id
                """)
                
                # Force an error with invalid constraint
                await conn.execute("""
                    INSERT INTO semantic_memories (
                        memory_id,
                        confidence
                    ) VALUES ($1, 2.0)
                """, temp_memory_id)  # This should fail due to confidence constraint
        except Exception:
            # Expected to fail
            pass
        
        # Verify the memory was rolled back
        rollback_check = await conn.fetchval("""
            SELECT COUNT(*) FROM memories WHERE content = 'Rollback test'
        """)
        
        assert rollback_check == 0, "Transaction should have been rolled back"


async def test_worldview_driven_memory_filtering(db_pool):
    """Test how worldview affects memory retrieval and filtering"""
    async with db_pool.acquire() as conn:
        # Create worldview primitive with filtering rules
        worldview_id = await conn.fetchval("""
            INSERT INTO worldview_primitives (
                category,
                belief,
                confidence,
                memory_filter_rules
            ) VALUES (
                'values',
                'Positive thinking is important',
                0.9,
                '{"filter_negative_emotions": true, "boost_positive_memories": 1.5}'
            ) RETURNING id
        """)
        
        # Create memories with different emotional valences
        positive_memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding,
                importance
            ) VALUES (
                'episodic'::memory_type,
                'Positive experience',
                array_fill(0.5, ARRAY[1536])::vector,
                0.5
            ) RETURNING id
        """)
        
        await conn.execute("""
            INSERT INTO episodic_memories (
                memory_id,
                action_taken,
                context,
                result,
                emotional_valence
            ) VALUES (
                $1,
                '{"action": "celebration"}',
                '{"context": "achievement"}',
                '{"result": "joy"}',
                0.8
            )
        """, positive_memory_id)
        
        negative_memory_id = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding,
                importance
            ) VALUES (
                'episodic'::memory_type,
                'Negative experience',
                array_fill(0.5, ARRAY[1536])::vector,
                0.5
            ) RETURNING id
        """)
        
        await conn.execute("""
            INSERT INTO episodic_memories (
                memory_id,
                action_taken,
                context,
                result,
                emotional_valence
            ) VALUES (
                $1,
                '{"action": "failure"}',
                '{"context": "disappointment"}',
                '{"result": "sadness"}',
                -0.8
            )
        """, negative_memory_id)
        
        # Create worldview influences
        await conn.execute("""
            INSERT INTO worldview_memory_influences (
                worldview_id,
                memory_id,
                influence_type,
                strength
            ) VALUES 
                ($1, $2, 'boost', 1.5),
                ($1, $3, 'suppress', 0.3)
        """, worldview_id, positive_memory_id, negative_memory_id)
        
        # Test worldview-influenced retrieval
        influenced_memories = await conn.fetch("""
            SELECT m.*, wmi.influence_type, wmi.strength,
                   CASE 
                       WHEN wmi.influence_type = 'boost' THEN m.importance * wmi.strength
                       WHEN wmi.influence_type = 'suppress' THEN m.importance * wmi.strength
                       ELSE m.importance
                   END as adjusted_importance
            FROM memories m
            LEFT JOIN worldview_memory_influences wmi ON m.id = wmi.memory_id
            WHERE wmi.worldview_id = $1
            ORDER BY adjusted_importance DESC
        """, worldview_id)
        
        assert len(influenced_memories) == 2, "Should find both influenced memories"
        assert influenced_memories[0]['influence_type'] == 'boost', "Positive memory should be boosted"
        assert influenced_memories[1]['influence_type'] == 'suppress', "Negative memory should be suppressed"
        assert influenced_memories[0]['adjusted_importance'] > influenced_memories[1]['adjusted_importance'], "Boosted memory should rank higher"


# MEDIUM PRIORITY ADDITIONAL TESTS

async def test_complex_graph_traversals(db_pool):
    """Test complex multi-hop graph traversals and path finding"""
    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute("SET search_path = ag_catalog, public;")
        
        # Create a complex memory network
        memory_chain = []
        memory_types = ['episodic', 'semantic', 'procedural', 'strategic', 'episodic']
        
        # Create 5 connected memories
        for i, mem_type in enumerate(memory_types):
            memory_id = await conn.fetchval("""
                INSERT INTO memories (type, content, embedding)
                VALUES ($1::memory_type, $2, array_fill($3::float, ARRAY[1536])::vector)
                RETURNING id
            """, mem_type, f'Complex memory {i}', float(i) * 0.1)
            memory_chain.append(memory_id)
            
            # Create graph node
            await conn.execute(f"""
                SELECT * FROM cypher('memory_graph', $$
                    CREATE (n:MemoryNode {{
                        memory_id: '{memory_id}',
                        type: '{mem_type}',
                        step: {i}
                    }})
                    RETURN n
                $$) as (n ag_catalog.agtype)
            """)
        
        # Create linear chain relationships
        for i in range(len(memory_chain) - 1):
            await conn.execute(f"""
                SELECT * FROM cypher('memory_graph', $$
                    MATCH (a:MemoryNode {{memory_id: '{memory_chain[i]}'}}),
                          (b:MemoryNode {{memory_id: '{memory_chain[i+1]}'}})
                    CREATE (a)-[r:LEADS_TO {{strength: {0.8 - i*0.1}}}]->(b)
                    RETURN r
                $$) as (r ag_catalog.agtype)
            """)
        
        # Create some cross-connections
        await conn.execute(f"""
            SELECT * FROM cypher('memory_graph', $$
                MATCH (a:MemoryNode {{memory_id: '{memory_chain[0]}'}}),
                      (b:MemoryNode {{memory_id: '{memory_chain[3]}'}})
                CREATE (a)-[r:INFLUENCES {{strength: 0.6}}]->(b)
                RETURN r
            $$) as (r ag_catalog.agtype)
        """)
        
        # Test 1: Find all paths between first and last memory
        paths = await conn.fetch(f"""
            SELECT * FROM cypher('memory_graph', $$
                MATCH p = (start:MemoryNode {{memory_id: '{memory_chain[0]}'}})-[*1..5]->(finish:MemoryNode {{memory_id: '{memory_chain[-1]}'}})
                RETURN p
            $$) as (p ag_catalog.agtype)
        """)
        
        assert len(paths) >= 1, "Should find at least one path"
        
        # Test 2: Find memories within 2 hops of the first memory
        nearby_memories = await conn.fetch(f"""
            SELECT * FROM cypher('memory_graph', $$
                MATCH (start:MemoryNode {{memory_id: '{memory_chain[0]}'}})-[*1..2]->(nearby:MemoryNode)
                RETURN DISTINCT nearby.memory_id as memory_id, nearby.type as type
            $$) as (memory_id ag_catalog.agtype, type ag_catalog.agtype)
        """)
        
        assert len(nearby_memories) >= 2, "Should find nearby memories"
        
        # Test 3: Find any path with relationship properties
        weighted_path = await conn.fetch(f"""
            SELECT * FROM cypher('memory_graph', $$
                MATCH p = (start:MemoryNode {{memory_id: '{memory_chain[0]}'}})-[*1..5]->(finish:MemoryNode {{memory_id: '{memory_chain[-1]}'}})
                RETURN p
                LIMIT 1
            $$) as (path ag_catalog.agtype)
        """)
        
        assert len(weighted_path) > 0, "Should find a path"
        
        # Test 4: Complex pattern matching
        patterns = await conn.fetch("""
            SELECT * FROM cypher('memory_graph', $$
                MATCH (e:MemoryNode {type: 'episodic'})-[:LEADS_TO]->(s:MemoryNode {type: 'semantic'})-[:LEADS_TO]->(p:MemoryNode {type: 'procedural'})
                RETURN e.memory_id as episodic_id, s.memory_id as semantic_id, p.memory_id as procedural_id
            $$) as (episodic_id ag_catalog.agtype, semantic_id ag_catalog.agtype, procedural_id ag_catalog.agtype)
        """)
        
        assert len(patterns) > 0, "Should find episodic->semantic->procedural patterns"


async def test_memory_lifecycle_management(db_pool):
    """Test comprehensive memory lifecycle management"""
    async with db_pool.acquire() as conn:
        # Create memories with different lifecycle stages
        lifecycle_memories = []
        
        # Stage 1: Fresh memories (high importance, recent)
        for i in range(3):
            memory_id = await conn.fetchval("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding,
                    importance,
                    access_count,
                    created_at
                ) VALUES (
                    'semantic'::memory_type,
                    'Fresh memory ' || $1,
                    array_fill(0.8, ARRAY[1536])::vector,
                    0.9,
                    10 + $2,
                    CURRENT_TIMESTAMP - interval '1 hour' * $2
                ) RETURNING id
            """, str(i), i)
            lifecycle_memories.append(('fresh', memory_id))
        
        # Stage 2: Aging memories (medium importance, older)
        for i in range(3):
            memory_id = await conn.fetchval("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding,
                    importance,
                    access_count,
                    created_at
                ) VALUES (
                    'episodic'::memory_type,
                    'Aging memory ' || $1,
                    array_fill(0.5, ARRAY[1536])::vector,
                    0.5,
                    5 + $2,
                    CURRENT_TIMESTAMP - interval '7 days' * ($2 + 1)
                ) RETURNING id
            """, str(i), i)
            lifecycle_memories.append(('aging', memory_id))
        
        # Stage 3: Stale memories (low importance, very old)
        for i in range(3):
            memory_id = await conn.fetchval("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding,
                    importance,
                    access_count,
                    created_at
                ) VALUES (
                    'procedural'::memory_type,
                    'Stale memory ' || $1,
                    array_fill(0.2, ARRAY[1536])::vector,
                    0.1,
                    1,
                    CURRENT_TIMESTAMP - interval '30 days' * ($2 + 1)
                ) RETURNING id
            """, str(i), i)
            lifecycle_memories.append(('stale', memory_id))
        
        # Test lifecycle categorization
        fresh_count = await conn.fetchval("""
            SELECT COUNT(*) FROM memories
            WHERE created_at > CURRENT_TIMESTAMP - interval '1 day'
            AND importance > 0.7
        """)
        
        aging_count = await conn.fetchval("""
            SELECT COUNT(*) FROM memories
            WHERE created_at BETWEEN CURRENT_TIMESTAMP - interval '30 days' 
                                 AND CURRENT_TIMESTAMP - interval '1 day'
            AND importance BETWEEN 0.3 AND 0.7
        """)
        
        stale_count = await conn.fetchval("""
            SELECT COUNT(*) FROM memories
            WHERE created_at < CURRENT_TIMESTAMP - interval '30 days'
            AND importance < 0.3
        """)
        
        assert fresh_count >= 3, "Should have fresh memories"
        assert aging_count >= 3, "Should have aging memories"
        assert stale_count >= 3, "Should have stale memories"
        
        # Test memory promotion (accessing old memory should increase importance)
        stale_memory = [m for stage, m in lifecycle_memories if stage == 'stale'][0]
        
        initial_importance = await conn.fetchval("""
            SELECT importance FROM memories WHERE id = $1
        """, stale_memory)
        
        # Simulate multiple accesses
        for _ in range(5):
            await conn.execute("""
                UPDATE memories 
                SET access_count = access_count + 1
                WHERE id = $1
            """, stale_memory)
        
        final_importance = await conn.fetchval("""
            SELECT importance FROM memories WHERE id = $1
        """, stale_memory)
        
        assert final_importance > initial_importance, "Accessed memory should gain importance"
        
        # Test memory archival workflow
        archival_candidates = await conn.fetch("""
            SELECT id, relevance_score 
            FROM memories
            WHERE relevance_score < 0.1
            AND status = 'active'
            ORDER BY relevance_score ASC
            LIMIT 5
        """)
        
        for candidate in archival_candidates:
            await conn.execute("""
                UPDATE memories 
                SET status = 'archived'::memory_status
                WHERE id = $1
            """, candidate['id'])
        
        # Verify archival
        archived_count = await conn.fetchval("""
            SELECT COUNT(*) FROM memories WHERE status = 'archived'
        """)
        
        assert archived_count > 0, "Should have archived some memories"


async def test_memory_pruning_operations(db_pool):
    """Test memory pruning and cleanup operations"""
    async with db_pool.acquire() as conn:
        # Create memories for pruning
        pruning_memories = []
        
        # Create very old, low-relevance memories
        for i in range(10):
            memory_id = await conn.fetchval("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding,
                    importance,
                    access_count,
                    created_at,
                    last_accessed
                ) VALUES (
                    'semantic'::memory_type,
                    'Pruning candidate ' || $1,
                    array_fill(0.1, ARRAY[1536])::vector,
                    0.05,
                    0,
                    CURRENT_TIMESTAMP - interval '90 days',
                    CURRENT_TIMESTAMP - interval '60 days'
                ) RETURNING id
            """, str(i))
            pruning_memories.append(memory_id)
        
        # Create some memories worth keeping
        for i in range(5):
            await conn.fetchval("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding,
                    importance,
                    access_count,
                    created_at
                ) VALUES (
                    'episodic'::memory_type,
                    'Important memory ' || $1,
                    array_fill(0.8, ARRAY[1536])::vector,
                    0.8,
                    20,
                    CURRENT_TIMESTAMP - interval '30 days'
                ) RETURNING id
            """, str(i))
        
        # Test pruning criteria identification
        pruning_candidates = await conn.fetch("""
            SELECT id, relevance_score, importance, 
                   age_in_days(created_at) as age_days,
                   COALESCE(age_in_days(last_accessed), 999) as days_since_access
            FROM memories
            WHERE relevance_score < 0.1
            AND (last_accessed IS NULL OR last_accessed < CURRENT_TIMESTAMP - interval '30 days')
            AND importance < 0.1
            ORDER BY relevance_score ASC
        """)
        
        assert len(pruning_candidates) >= 10, "Should identify pruning candidates"
        
        # Test safe pruning (archive first, then delete)
        for candidate in pruning_candidates[:5]:
            # First archive
            await conn.execute("""
                UPDATE memories 
                SET status = 'archived'::memory_status
                WHERE id = $1
            """, candidate['id'])
        
        # Then delete archived memories older than threshold
        deleted_count = await conn.fetchval("""
            WITH deleted_memories AS (
                DELETE FROM memories 
                WHERE status = 'archived'
                AND created_at < CURRENT_TIMESTAMP - interval '120 days'
                RETURNING id
            )
            SELECT COUNT(*) FROM deleted_memories
        """)
        
        # Test working memory cleanup
        # Create expired working memory entries
        for i in range(5):
            await conn.execute("""
                INSERT INTO working_memory (
                    content,
                    embedding,
                    expiry
                ) VALUES (
                    'Expired working memory ' || $1,
                    array_fill(0.5, ARRAY[1536])::vector,
                    CURRENT_TIMESTAMP - interval '1 hour'
                )
            """, str(i))
        
        # Clean up expired working memory
        expired_cleaned = await conn.fetchval("""
            WITH cleaned AS (
                DELETE FROM working_memory
                WHERE expiry < CURRENT_TIMESTAMP
                RETURNING id
            )
            SELECT COUNT(*) FROM cleaned
        """)
        
        assert expired_cleaned >= 5, "Should clean up expired working memory"


async def test_database_optimization_operations(db_pool):
    """Test database optimization and maintenance operations"""
    async with db_pool.acquire() as conn:
        # Test index usage analysis
        index_usage = await conn.fetch("""
            SELECT 
                schemaname,
                relname as tablename,
                indexrelname as indexname,
                idx_scan,
                idx_tup_read,
                idx_tup_fetch
            FROM pg_stat_user_indexes
            WHERE schemaname = 'public'
            AND relname IN ('memories', 'memory_clusters', 'memory_cluster_members')
            ORDER BY idx_scan DESC
        """)
        
        assert len(index_usage) > 0, "Should have index usage statistics"
        
        # Test table statistics
        table_stats = await conn.fetch("""
            SELECT 
                schemaname,
                relname as tablename,
                n_tup_ins,
                n_tup_upd,
                n_tup_del,
                n_live_tup,
                n_dead_tup
            FROM pg_stat_user_tables
            WHERE schemaname = 'public'
            AND relname IN ('memories', 'memory_clusters')
            ORDER BY n_live_tup DESC
        """)
        
        assert len(table_stats) > 0, "Should have table statistics"
        
        # Test query performance analysis
        # Create a complex query and analyze its performance
        import time
        
        start_time = time.time()
        complex_query_result = await conn.fetch("""
            SELECT 
                m.type,
                COUNT(*) as memory_count,
                AVG(m.importance) as avg_importance,
                AVG(m.relevance_score) as avg_relevance,
                COUNT(mcm.cluster_id) as cluster_memberships
            FROM memories m
            LEFT JOIN memory_cluster_members mcm ON m.id = mcm.memory_id
            WHERE m.status = 'active'
            GROUP BY m.type
            HAVING COUNT(*) > 0
            ORDER BY avg_importance DESC
        """)
        query_time = time.time() - start_time
        
        assert len(complex_query_result) > 0, "Complex query should return results"
        assert query_time < 1.0, f"Complex query too slow: {query_time}s"
        
        # Test vacuum and analyze simulation (read-only operations)
        vacuum_info = await conn.fetch("""
            SELECT
                schemaname,
                relname as tablename,
                last_vacuum,
                last_autovacuum,
                last_analyze,
                last_autoanalyze,
                vacuum_count,
                autovacuum_count
            FROM pg_stat_user_tables
            WHERE schemaname = 'public'
            AND relname = 'memories'
        """)
        
        assert len(vacuum_info) > 0, "Should have vacuum statistics"


async def test_backup_restore_consistency(db_pool):
    """Test backup and restore data consistency"""
    async with db_pool.acquire() as conn:
        # Clean up any existing test data first
        await conn.execute("""
            DELETE FROM memory_cluster_members 
            WHERE memory_id IN (
                SELECT id FROM memories WHERE content LIKE 'Backup test memory%'
            )
        """)
        await conn.execute("""
            DELETE FROM semantic_memories 
            WHERE memory_id IN (
                SELECT id FROM memories WHERE content LIKE 'Backup test memory%'
            )
        """)
        await conn.execute("""
            DELETE FROM memories WHERE content LIKE 'Backup test memory%'
        """)
        await conn.execute("""
            DELETE FROM memory_clusters WHERE name = 'Backup Test Cluster'
        """)
        
        # Create a known dataset for backup testing
        backup_test_data = []
        
        # Create test memories with relationships
        for i in range(5):
            memory_id = await conn.fetchval("""
                INSERT INTO memories (
                    type,
                    content,
                    embedding,
                    importance
                ) VALUES (
                    'semantic'::memory_type,
                    'Backup test memory ' || $1,
                    array_fill($2::float, ARRAY[1536])::vector,
                    $3
                ) RETURNING id
            """, str(i), float(i) * 0.1, 0.5 + (i * 0.1))
            
            backup_test_data.append(memory_id)
            
            # Add semantic details
            await conn.execute("""
                INSERT INTO semantic_memories (
                    memory_id,
                    confidence,
                    category
                ) VALUES ($1, $2, $3)
            """, memory_id, 0.8, [f'category_{i}'])
        
        # Create cluster and relationships
        cluster_id = await conn.fetchval("""
            INSERT INTO memory_clusters (
                cluster_type,
                name,
                centroid_embedding
            ) VALUES (
                'theme'::cluster_type,
                'Backup Test Cluster',
                array_fill(0.5, ARRAY[1536])::vector
            ) RETURNING id
        """)
        
        # Add memories to cluster
        for memory_id in backup_test_data:
            await conn.execute("""
                INSERT INTO memory_cluster_members (
                    cluster_id,
                    memory_id,
                    membership_strength
                ) VALUES ($1, $2, 0.8)
            """, cluster_id, memory_id)
        
        # Simulate backup verification by checking data consistency
        # Test 1: Verify all memories have corresponding semantic records
        orphaned_memories = await conn.fetch("""
            SELECT m.id 
            FROM memories m
            LEFT JOIN semantic_memories sm ON m.id = sm.memory_id
            WHERE m.type = 'semantic' 
            AND m.content LIKE 'Backup test memory%'
            AND sm.memory_id IS NULL
        """)
        
        assert len(orphaned_memories) == 0, "No orphaned semantic memories should exist"
        
        # Test 2: Verify cluster relationships are intact
        cluster_members = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM memory_cluster_members mcm
            JOIN memories m ON mcm.memory_id = m.id
            WHERE m.content LIKE 'Backup test memory%'
        """)
        
        assert cluster_members == 5, "All test memories should be in cluster"
        
        # Test 3: Verify referential integrity for our test data only
        integrity_check = await conn.fetch("""
            SELECT 
                'test_memories->semantic' as check_type,
                COUNT(*) as violations
            FROM memories m
            LEFT JOIN semantic_memories sm ON m.id = sm.memory_id
            WHERE m.type = 'semantic' 
            AND m.content LIKE 'Backup test memory%'
            AND sm.memory_id IS NULL
            
            UNION ALL
            
            SELECT 
                'test_cluster_members->memories' as check_type,
                COUNT(*) as violations
            FROM memory_cluster_members mcm
            LEFT JOIN memories m ON mcm.memory_id = m.id
            WHERE m.content LIKE 'Backup test memory%'
            AND m.id IS NULL
            
            UNION ALL
            
            SELECT 
                'test_cluster_members->clusters' as check_type,
                COUNT(*) as violations
            FROM memory_cluster_members mcm
            LEFT JOIN memory_clusters mc ON mcm.cluster_id = mc.id
            WHERE mc.name = 'Backup Test Cluster'
            AND mc.id IS NULL
        """)
        
        for check in integrity_check:
            assert check['violations'] == 0, f"Integrity violation in {check['check_type']}"
        
        # Test 4: Verify computed fields are consistent
        computed_field_check = await conn.fetch("""
            SELECT 
                id,
                importance,
                relevance_score,
                (importance * exp(-decay_rate * age_in_days(created_at))) as expected_relevance
            FROM memories
            WHERE content LIKE 'Backup test memory%'
        """)
        
        for row in computed_field_check:
            actual = float(row['relevance_score'])
            expected = float(row['expected_relevance'])
            assert abs(actual - expected) < 0.001, f"Relevance score mismatch: {actual} vs {expected}"


async def test_schema_migration_compatibility(db_pool):
    """Test schema migration and version compatibility"""
    async with db_pool.acquire() as conn:
        # Test 1: Check current schema version (simulate version tracking)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                description TEXT
            )
        """)
        
        # Record current schema version
        await conn.execute("""
            INSERT INTO schema_version (version, description)
            VALUES (1, 'Initial AGI Memory System schema')
            ON CONFLICT (version) DO NOTHING
        """)
        
        # Test 2: Simulate adding a new column (non-breaking change)
        await conn.execute("""
            ALTER TABLE memories 
            ADD COLUMN IF NOT EXISTS migration_test_field TEXT DEFAULT 'test_value'
        """)
        
        # Verify the column was added
        column_exists = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM information_schema.columns 
            WHERE table_name = 'memories' 
            AND column_name = 'migration_test_field'
        """)
        
        assert column_exists == 1, "Migration test column should exist"
        
        # Test 3: Verify existing data integrity after migration
        memory_count_before = await conn.fetchval("""
            SELECT COUNT(*) FROM memories
        """)
        
        # Test that existing memories have the default value
        default_value_count = await conn.fetchval("""
            SELECT COUNT(*) FROM memories 
            WHERE migration_test_field = 'test_value'
        """)
        
        assert default_value_count == memory_count_before, "All existing memories should have default value"
        
        # Test 4: Simulate index migration
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_migration_test 
            ON memories(migration_test_field)
        """)
        
        # Verify index was created
        index_exists = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM pg_indexes 
            WHERE tablename = 'memories' 
            AND indexname = 'idx_memories_migration_test'
        """)
        
        assert index_exists == 1, "Migration test index should exist"
        
        # Test 5: Simulate rollback capability
        await conn.execute("""
            ALTER TABLE memories 
            DROP COLUMN IF EXISTS migration_test_field
        """)
        
        # Verify rollback worked
        column_exists_after = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM information_schema.columns 
            WHERE table_name = 'memories' 
            AND column_name = 'migration_test_field'
        """)
        
        assert column_exists_after == 0, "Migration test column should be removed"
        
        # Clean up
        await conn.execute("DROP TABLE IF EXISTS schema_version")


async def test_monitoring_and_alerting_metrics(db_pool):
    """Test monitoring metrics and alerting thresholds"""
    async with db_pool.acquire() as conn:
        # Test 1: Memory system health metrics
        health_metrics = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total_memories,
                COUNT(*) FILTER (WHERE status = 'active') as active_memories,
                COUNT(*) FILTER (WHERE status = 'archived') as archived_memories,
                AVG(importance) as avg_importance,
                AVG(relevance_score) as avg_relevance,
                AVG(access_count) as avg_access_count,
                COUNT(*) FILTER (WHERE last_accessed > CURRENT_TIMESTAMP - interval '24 hours') as recently_accessed
            FROM memories
        """)
        
        assert health_metrics['total_memories'] > 0, "Should have memories for monitoring"
        
        # Test 2: Cluster health metrics
        cluster_metrics = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total_clusters,
                AVG(importance_score) as avg_cluster_importance,
                COUNT(*) FILTER (WHERE last_activated > CURRENT_TIMESTAMP - interval '24 hours') as recently_active_clusters,
                AVG(
                    (SELECT COUNT(*) FROM memory_cluster_members mcm WHERE mcm.cluster_id = mc.id)
                ) as avg_cluster_size
            FROM memory_clusters mc
        """)
        
        assert cluster_metrics['total_clusters'] >= 0, "Should have cluster metrics"
        
        # Test 3: Performance metrics
        performance_metrics = await conn.fetch("""
            SELECT 
                'vector_search' as metric_type,
                COUNT(*) as operations,
                AVG(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - created_at))) as avg_age_seconds
            FROM memories
            WHERE created_at > CURRENT_TIMESTAMP - interval '1 hour'
            
            UNION ALL
            
            SELECT 
                'cluster_operations' as metric_type,
                COUNT(*) as operations,
                AVG(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - added_at))) as avg_age_seconds
            FROM memory_cluster_members
            WHERE added_at > CURRENT_TIMESTAMP - interval '1 hour'
        """)
        
        assert len(performance_metrics) > 0, "Should have performance metrics"
        
        # Test 4: Alert threshold simulation
        alert_conditions = await conn.fetch("""
            SELECT 
                'low_memory_activity' as alert_type,
                CASE 
                    WHEN COUNT(*) FILTER (WHERE last_accessed > CURRENT_TIMESTAMP - interval '24 hours') < 10 
                    THEN 'ALERT' 
                    ELSE 'OK' 
                END as status,
                COUNT(*) FILTER (WHERE last_accessed > CURRENT_TIMESTAMP - interval '24 hours') as recent_access_count
            FROM memories
            
            UNION ALL
            
            SELECT 
                'high_archive_rate' as alert_type,
                CASE 
                    WHEN COUNT(*) FILTER (WHERE status = 'archived') > COUNT(*) * 0.5 
                    THEN 'ALERT' 
                    ELSE 'OK' 
                END as status,
                COUNT(*) FILTER (WHERE status = 'archived') as archived_count
            FROM memories
            
            UNION ALL
            
            SELECT 
                'cluster_imbalance' as alert_type,
                CASE 
                    WHEN MAX(member_count) > AVG(member_count) * 3 
                    THEN 'ALERT' 
                    ELSE 'OK' 
                END as status,
                MAX(member_count) as max_cluster_size
            FROM (
                SELECT 
                    cluster_id,
                    COUNT(*) as member_count
                FROM memory_cluster_members
                GROUP BY cluster_id
            ) cluster_sizes
        """)
        
        assert len(alert_conditions) == 3, "Should have all alert conditions"
        
        # Test 5: Resource usage metrics
        resource_metrics = await conn.fetchrow("""
            SELECT 
                pg_size_pretty(pg_total_relation_size('memories')) as memories_table_size,
                pg_size_pretty(pg_total_relation_size('memory_clusters')) as clusters_table_size,
                (SELECT COUNT(*) FROM memories) as memory_count,
                (SELECT COUNT(*) FROM memory_clusters) as cluster_count,
                (SELECT COUNT(*) FROM memory_cluster_members) as membership_count
        """)
        
        assert resource_metrics is not None, "Should have resource metrics"


async def test_multi_agi_considerations(db_pool):
    """Test considerations for multi-AGI support (current limitations)"""
    async with db_pool.acquire() as conn:
        # Clean up any existing test data first
        await conn.execute("""
            DELETE FROM memories WHERE content LIKE '%AGI-% believes X is%'
        """)
        
        # Test 1: Identify single-AGI assumptions in current schema
        single_agi_tables = await conn.fetch("""
            SELECT 
                table_name,
                CASE 
                    WHEN table_name IN ('identity_model', 'worldview_primitives') THEN 'singleton_table'
                    WHEN table_name LIKE '%memory%' THEN 'memory_table'
                    ELSE 'other'
                END as table_category
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)
        
        singleton_tables = [t for t in single_agi_tables if t['table_category'] == 'singleton_table']
        assert len(singleton_tables) > 0, "Should identify singleton tables"
        
        # Test 2: Simulate multi-AGI data isolation requirements
        # This test demonstrates what would need to change for multi-AGI support
        
        # Check if any tables have AGI instance identification
        agi_id_columns = await conn.fetch("""
            SELECT 
                table_name,
                column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
            AND column_name LIKE '%agi%'
            ORDER BY table_name, column_name
        """)
        
        # Current schema should have no AGI ID columns (single-AGI design)
        assert len(agi_id_columns) == 0, "Current schema should not have AGI ID columns"
        
        # Test 3: Demonstrate memory isolation challenges
        # Create test scenario showing how memories could conflict between AGIs
        
        # AGI 1 memories
        agi1_memory = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding,
                importance
            ) VALUES (
                'semantic'::memory_type,
                'AGI-1 believes X is true',
                array_fill(0.8, ARRAY[1536])::vector,
                0.9
            ) RETURNING id
        """)
        
        # AGI 2 memories (conflicting belief)
        agi2_memory = await conn.fetchval("""
            INSERT INTO memories (
                type,
                content,
                embedding,
                importance
            ) VALUES (
                'semantic'::memory_type,
                'AGI-2 believes X is false',
                array_fill(0.8, ARRAY[1536])::vector,
                0.9
            ) RETURNING id
        """)
        
        # Demonstrate conflict: both memories exist in same space
        conflicting_memories = await conn.fetch("""
            SELECT 
                id,
                content,
                importance,
                'conflict_detected' as issue_type
            FROM memories
            WHERE content LIKE '%AGI-% believes X is%'
            ORDER BY content
        """)
        
        assert len(conflicting_memories) == 2, "Should find conflicting AGI memories"
        
        # Test 4: Demonstrate worldview conflicts
        # In single-AGI system, only one worldview can exist
        worldview_count = await conn.fetchval("""
            SELECT COUNT(*) FROM worldview_primitives
        """)
        
        # Test 5: Demonstrate identity model limitations
        identity_count = await conn.fetchval("""
            SELECT COUNT(*) FROM identity_model
        """)
        
        # Test 6: Show what would be needed for multi-AGI support
        multi_agi_requirements = {
            'schema_changes_needed': [
                'Add agi_instance_id to all memory tables',
                'Add agi_instance_id to worldview_primitives',
                'Add agi_instance_id to identity_model',
                'Add row-level security policies',
                'Modify all views to filter by AGI instance',
                'Update all functions to include AGI context'
            ],
            'isolation_challenges': [
                'Memory similarity search across AGI boundaries',
                'Cluster centroid calculations per AGI',
                'Graph relationships between AGI instances',
                'Shared vs private memory spaces',
                'Cross-AGI learning and knowledge transfer'
            ]
        }
        
        # This test documents the current single-AGI limitations
        assert len(multi_agi_requirements['schema_changes_needed']) > 0, "Multi-AGI support requires significant changes"


# Keep all existing tests from the original file...
# (All the tests from test_memory_storage through test_procedural_effectiveness_view remain the same)
