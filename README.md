# AGI Memory System

A sophisticated database design for Artificial General Intelligence (AGI) memory management, implementing multiple types of memory storage and retrieval mechanisms inspired by human cognitive architecture.

## Overview

This system provides a comprehensive memory management solution for AGI applications, featuring:

- Multiple memory types (Episodic, Semantic, Procedural, Strategic)
- Vector-based memory storage and similarity search
- Graph-based memory relationships
- Dynamic memory importance calculation
- Memory decay simulation
- Working memory system
- Memory consolidation mechanisms

## Architecture

### Memory Types

1. **Working Memory**
   - Temporary storage for active processing
   - Automatic expiry mechanism
   - Vector embeddings for content similarity

2. **Episodic Memory**
   - Event-based memories with temporal context
   - Stores actions, contexts, and results
   - Emotional valence tracking
   - Verification status

3. **Semantic Memory**
   - Fact-based knowledge storage
   - Confidence scoring
   - Source tracking
   - Contradiction management
   - Categorical organization

4. **Procedural Memory**
   - Step-by-step procedure storage
   - Success rate tracking
   - Duration monitoring
   - Failure point analysis

5. **Strategic Memory**
   - Pattern recognition storage
   - Adaptation history
   - Context applicability
   - Success metrics

### Advanced Features

**Memory Clustering:**
- Automatic thematic grouping of related memories
- Emotional signature tracking
- Cross-cluster relationship mapping
- Activation pattern analysis

**Worldview Integration:**
- Belief system modeling with confidence scores
- Memory filtering based on worldview alignment
- Identity-core memory cluster identification
- Adaptive memory importance based on beliefs

**Graph Relationships:**
- Apache AGE integration for complex memory networks
- Multi-hop relationship traversal
- Pattern detection across memory types
- Causal relationship modeling

### Key Features

- **Vector Embeddings**: Uses pgvector for similarity-based memory retrieval
- **Graph Relationships**: Apache AGE integration for complex memory relationships
- **Dynamic Scoring**: Automatic calculation of memory importance and relevance
- **Memory Decay**: Time-based decay simulation for realistic memory management
- **Change Tracking**: Historical tracking of memory modifications

## Technical Stack

- **Database**: PostgreSQL with extensions:
  - pgvector (vector similarity)
  - AGE (graph database)
  - btree_gist
  - pg_trgm
  - cube

## Dependencies

### Python Requirements
- asyncpg>=0.29.0 (PostgreSQL async driver)
- pytest>=7.4.3 (testing framework)
- numpy>=1.24.0 (numerical operations)
- fastapi>=0.104.0 (web framework)
- pydantic>=2.4.2 (data validation)

### Node.js Requirements  
- @modelcontextprotocol/sdk (MCP framework)
- pg (PostgreSQL driver)

### Database Extensions
- pgvector (vector similarity)
- AGE (graph database)
- pg_trgm (text search)
- btree_gist (indexing)
- cube (multidimensional indexing)

## Environment Configuration

Copy `.env.local` to `.env` and configure:

```bash
POSTGRES_DB=agi_db           # Database name
POSTGRES_USER=agi_user       # Database user
POSTGRES_PASSWORD=agi_password # Database password
```

For MCP server, also set:
```bash
POSTGRES_HOST=localhost      # Database host
POSTGRES_PORT=5432          # Database port
```

## Setup

```bash
cp .env.local .env # modify the .env file with your own values
docker compose up -d
```

This will:
1. Start a PostgreSQL instance with all required extensions (pgvector, AGE, etc.)
2. Initialize the database schema
3. Create necessary tables, functions, and triggers

## Testing

Run the test suite with:

```bash
pytest test.py -v
```

## API Reference (MCP Tools)

### Memory Operations
- `create_memory(type, content, embedding, importance, metadata)` - Create new memories
- `get_memory(memory_id)` - Retrieve and access specific memory
- `search_memories_similarity(embedding, limit, threshold)` - Vector similarity search
- `search_memories_text(query, limit)` - Full-text search

### Cluster Operations  
- `get_memory_clusters(limit)` - List memory clusters by importance
- `activate_cluster(cluster_id, context)` - Activate cluster and get memories
- `create_memory_cluster(name, type, description, keywords)` - Create new cluster

### System Introspection
- `get_identity_core()` - Retrieve identity model and core clusters
- `get_worldview()` - Get worldview primitives and beliefs
- `get_memory_health()` - System health statistics
- `get_active_themes(days)` - Recently activated themes

## Usage Examples

### Creating Memories
```python
# Via MCP tools
memory = await create_memory(
    type="episodic",
    content="User expressed interest in machine learning",
    embedding=embedding_vector,
    importance=0.8,
    metadata={
        "emotional_valence": 0.6,
        "context": {"topic": "AI", "user_mood": "curious"}
    }
)
```

### Searching Memories
```python
# Similarity search
similar = await search_memories_similarity(
    embedding=query_vector,
    limit=10,
    threshold=0.7
)

# Text search
results = await search_memories_text(
    query="machine learning concepts",
    limit=5
)
```

## Database Schema

### Core Tables
1. **working_memory**
   - Temporary storage with automatic expiry
   - Vector embeddings for similarity search
   - Priority scoring for attention mechanisms

2. **memories**
   - Permanent storage for consolidated memories
   - Links to specific memory type tables
   - Metadata tracking (creation, modification, access)

3. **memory_relationships**
   - Graph-based relationship storage
   - Bidirectional links between memories
   - Relationship type classification

### Memory Type Tables
Each specialized memory type has its own table with type-specific fields:
- episodic_memories
- semantic_memories
- procedural_memories
- strategic_memories

### Clustering Tables
- memory_clusters
- memory_cluster_members
- cluster_relationships
- cluster_activation_history

### Identity and Worldview Tables
- identity_model
- worldview_primitives
- worldview_memory_influences
- identity_memory_resonance

### Indexes and Constraints
- Vector indexes for similarity search
- Graph indexes for relationship traversal
- Temporal indexes for time-based queries

## Example Queries

### Memory Retrieval
```sql
-- Find similar memories using vector similarity
SELECT * FROM memories
WHERE embedding <-> query_embedding < threshold
ORDER BY embedding <-> query_embedding
LIMIT 10;

-- Find related memories through graph
SELECT * FROM ag_catalog.cypher('memory_graph', $$
    MATCH (m:MemoryNode)-[:RELATES_TO]->(related)
    WHERE m.id = $memory_id
    RETURN related
$$) as (related agtype);
```

## Performance Characteristics

- **Vector Search**: Sub-second similarity queries on 10K+ memories
- **Memory Storage**: Supports millions of memories with proper indexing
- **Cluster Operations**: Efficient graph traversal for relationship queries
- **Maintenance**: Requires periodic consolidation and pruning

### Scaling Considerations
- Memory consolidation recommended every 4-6 hours
- Database optimization during off-peak hours
- Monitor vector index performance with large datasets

## System Maintenance

The memory system requires three key maintenance processes to function effectively:

### 1. Memory Consolidation
Short-term memories need to be consolidated into long-term storage. This process should:
- Move frequently accessed items from working memory to permanent storage
- Run periodically (recommended every 4-6 hours)
- Consider memory importance and access patterns

### 2. Memory Pruning
The system needs regular cleanup to prevent overwhelming storage:
- Archive or remove low-relevance memories
- Decay importance scores of unused memories
- Run daily or weekly, depending on system usage

### 3. Database Optimization
Regular database maintenance ensures optimal performance:
- Reindex tables for efficient vector searches
- Update statistics for query optimization
- Run during off-peak hours

### Implementation Note
These maintenance tasks can be implemented using:
- Database scheduling tools
- External task schedulers
- System-level scheduling (cron, systemd, etc.)

Choose the scheduling method that best fits your infrastructure and monitoring capabilities. Ensure proper logging and error handling for all maintenance operations.

## Troubleshooting

### Common Issues

**Database Connection Errors:**
- Ensure PostgreSQL is running: `docker compose ps`
- Check logs: `docker compose logs db`
- Verify extensions: Run test suite with `pytest test.py -v`

**Memory Search Performance:**
- Rebuild vector indexes if queries are slow
- Check memory_health view for system statistics
- Consider memory pruning if dataset is very large

**MCP Server Issues:**
- Verify Node.js dependencies: `npm install`
- Check database connectivity from MCP server
- Ensure environment variables are set correctly

## Usage Guide

### Memory Interaction Flow

The AGI Memory System provides a layered approach to memory management, similar to human cognitive processes:

1. **Initial Memory Creation**
   - New information enters through working memory
   - System assigns initial importance scores
   - Vector embeddings are generated for similarity matching

2. **Memory Retrieval**
   - Query across multiple memory types simultaneously
   - Use similarity search for related memories
   - Access through graph relationships for connected concepts

3. **Memory Updates**
   - Automatic tracking of memory modifications
   - Importance scores adjust based on usage
   - Relationships update dynamically

4. **Memory Integration**
   - Cross-referencing between memory types
   - Automatic relationship discovery
   - Pattern recognition across memories

```mermaid
graph TD
    Input[New Information] --> WM[Working Memory]
    WM --> |Consolidation| LTM[Long-Term Memory]
    
    subgraph "Long-Term Memory"
        LTM --> EM[Episodic Memory]
        LTM --> SM[Semantic Memory]
        LTM --> PM[Procedural Memory]
        LTM --> STM[Strategic Memory]
    end
    
    Query[Query/Retrieval] --> |Vector Search| LTM
    Query --> |Graph Traversal| LTM
    
    EM ---|Relationships| SM
    SM ---|Relationships| PM
    PM ---|Relationships| STM
    
    LTM --> |Decay| Archive[Archive/Removal]
    WM --> |Cleanup| Archive
```

### Key Integration Points

- Use the MCP API for all memory operations
- Implement proper error handling for failed operations
- Monitor memory usage and system performance
- Regular backup of critical memories

### Best Practices

- Initialize working memory with reasonable size limits
- Implement rate limiting for memory operations
- Regular validation of memory consistency
- Monitor and adjust importance scoring parameters

## Important Note

This database schema is designed for a single AGI instance. Supporting multiple AGI instances would require significant schema modifications, including:

- Adding AGI instance identification to all memory tables
- Partitioning strategies for memory isolation
- Modified relationship handling for cross-AGI memory sharing
- Separate working memory spaces per AGI
- Additional access controls and memory ownership

If you need multi-AGI support, consider refactoring the schema to include tenant isolation patterns before implementation.
