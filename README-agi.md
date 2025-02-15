# AGI Memory System: Comprehensive Documentation

This document provides a comprehensive explanation of the AGI Memory System, a sophisticated database design for Artificial General Intelligence (AGI) memory management.

## Introduction

The AGI Memory System is designed to provide a robust and flexible memory management solution for AGI applications. It draws inspiration from human cognitive architecture, implementing multiple memory types and mechanisms for storing, retrieving, and managing memories. This system aims to enable AGIs to effectively store, access, and reason about information, mimicking the way humans learn and remember.

## Overview

The core concept of the AGI Memory System is to provide a structured and efficient way to manage the vast amounts of information an AGI might encounter.  It goes beyond simple data storage by incorporating features that allow the AGI to:

*   **Categorize Information:** Store different types of information in specialized memory stores (episodic, semantic, procedural, strategic).
*   **Find Relationships:** Connect memories through a graph database, representing relationships and dependencies.
*   **Prioritize Information:** Dynamically calculate the importance and relevance of memories based on usage and context.
*   **Simulate Forgetting:** Implement memory decay to simulate the natural forgetting process.
*   **Track Changes:** Maintain a history of memory modifications for auditing and analysis.
* **Integrate Worldview and Identity:** Incorporate worldview primitives and an identity model to influence memory processing and retrieval.

## Architecture

### Memory Types

The system implements five primary memory types, each designed to store a specific kind of information:

1.  **Working Memory:**
    *   **Purpose:** Temporary storage for information actively being processed.  Similar to short-term memory in humans.
    *   **Fields:**
        *   `id` (UUID): Unique identifier.
        *   `content` (TEXT): The content of the memory.
        *   `embedding` (VECTOR): Vector representation for similarity search.
        *   `expiry` (TIMESTAMP): Time when the memory should expire.
        *   `importance` (REAL): Initial importance score.
        *   `access_count` (INTEGER): Number of times accessed.
        *   `relevance_score` (REAL): Calculated relevance score.
    *   **Mechanism:** Automatic expiry mechanism removes outdated information.

2.  **Episodic Memory:**
    *   **Purpose:** Stores event-based memories, including actions, contexts, and results.  Represents specific experiences.
    *   **Fields (in `episodic_memories` table):**
        *   `memory_id` (UUID): Foreign key referencing the `memories` table.
        *   `action_taken` (JSONB):  Details of the action taken.
        *   `context` (JSONB):  The context surrounding the event.
        *   `result` (JSONB):  The outcome of the action.
        *   `emotional_valence` (REAL):  Emotional impact of the event.
        *   `verification_status` (BOOLEAN): Whether the memory has been verified.
        *   `event_time` (TIMESTAMP): Timestamp of the event.
    *   **Mechanism:**  Provides a chronological record of the AGI's experiences.

3.  **Semantic Memory:**
    *   **Purpose:** Stores fact-based knowledge. Represents general knowledge about the world.
    *   **Fields (in `semantic_memories` table):**
        *   `memory_id` (UUID): Foreign key referencing the `memories` table.
        *   `confidence` (REAL):  Confidence level in the fact's accuracy.
        *   `source_references` (JSONB):  Sources of the information.
        *   `contradictions` (JSONB):  Information that contradicts this memory.
        *   `category` (TEXT[]): Categories the fact belongs to.
        *   `related_concepts` (TEXT[]): Related concepts.
        *   `last_validated` (TIMESTAMP): Timestamp of last validation.
    *   **Mechanism:**  Organizes knowledge categorically and tracks confidence and sources.

4.  **Procedural Memory:**
    *   **Purpose:** Stores step-by-step procedures or skills.  Represents "how-to" knowledge.
    *   **Fields (in `procedural_memories` table):**
        *   `memory_id` (UUID): Foreign key referencing the `memories` table.
        *   `steps` (JSONB):  The sequence of steps in the procedure.
        *   `prerequisites` (JSONB):  Conditions required before executing the procedure.
        *   `success_count` (INTEGER): Number of successful executions.
        *   `total_attempts` (INTEGER): Total number of attempts.
        *   `average_duration` (INTERVAL): Average execution time.
        *   `failure_points` (JSONB):  Information about points of failure.
        *   `success_rate` (REAL): Calculated success rate.
    *   **Mechanism:**  Tracks success rates and failure points to improve procedure execution.

5.  **Strategic Memory:**
    *   **Purpose:** Stores patterns, strategies, and adaptations. Represents higher-level planning knowledge.
    *   **Fields (in `strategic_memories` table):**
        *   `memory_id` (UUID): Foreign key referencing the `memories` table.
        *   `pattern_description` (TEXT): Description of the strategic pattern.
        *   `supporting_evidence` (JSONB): Evidence supporting the strategy.
        *   `confidence_score` (REAL): Confidence in the strategy's effectiveness.
        *   `success_metrics` (JSONB): Metrics for evaluating success.
        *   `adaptation_history` (JSONB): Record of adaptations made to the strategy.
        *   `context_applicability` (JSONB): Contexts where the strategy is applicable.
    *   **Mechanism:**  Stores and refines strategies based on experience and context.

### Key Features

*   **Vector Embeddings:** Uses `pgvector` to store vector representations of memories, enabling similarity-based retrieval. This allows the AGI to find memories that are semantically similar to a given query, even if they don't share exact keywords.
*   **Graph Relationships:** Integrates Apache AGE to represent complex relationships between memories. This allows the AGI to navigate connections between different pieces of information, forming a network of knowledge.
*   **Dynamic Scoring:** Automatically calculates the importance and relevance of memories based on factors like access frequency, recency, and relationships to other memories.
*   **Memory Decay:** Simulates time-based decay of memory importance, reflecting the natural forgetting process. This helps to prioritize relevant memories and manage storage space.
*   **Change Tracking:** Maintains a historical record of memory modifications, allowing the AGI to track how its knowledge has evolved over time.
*   **Worldview Primitives:** Incorporates `worldview_primitives` to represent fundamental beliefs and values, influencing memory filtering and interpretation.
*   **Identity Model:** Includes an `identity_model` to represent the AGI's self-concept and agency beliefs, affecting memory resonance and integration.

## Technical Stack

*   **Database:** PostgreSQL
*   **Extensions:**
    *   `pgvector`: For vector similarity search.
    *   Apache AGE: For graph database functionality.
    *   `btree_gist`: For GiST index support.
    *   `pg_trgm`: For trigram-based text similarity.
    *   `cube`: For multidimensional cube data type.

## Setup

1.  **Environment Variables:**
    *   Copy the `.env.local` file to `.env`.
    *   Modify the `.env` file to set your desired environment variables (e.g., database credentials).

2.  **Docker Compose:**
    *   Run `docker compose up -d` to start the PostgreSQL instance with all required extensions. This will also initialize the database schema.

## Database Schema

### Core Tables

1.  **`working_memory`:**
    *   `id` (UUID, PRIMARY KEY): Unique identifier.
    *   `content` (TEXT): Memory content.
    *   `embedding` (VECTOR): Vector embedding.
    *   `expiry` (TIMESTAMP): Expiry timestamp.
    *   `importance` (REAL): Initial importance.
    *   `access_count` (INTEGER): Access count.
    *   `relevance_score` (REAL): Calculated relevance.

2.  **`memories`:**
    *   `id` (UUID, PRIMARY KEY): Unique identifier.
    *   `type` (memory_type ENUM): Type of memory ('episodic', 'semantic', 'procedural', 'strategic').
    *   `content` (TEXT, NOT NULL): Memory content.
    *   `embedding` (VECTOR): Vector embedding.
    *   `importance` (REAL): Importance score.
    *   `decay_rate` (REAL): Decay rate.
    *   `created_at` (TIMESTAMP): Creation timestamp.
    *   `updated_at` (TIMESTAMP): Last update timestamp.
    *   `last_accessed` (TIMESTAMP): Last access timestamp.
    *   `access_count` (INTEGER): Access count.
    *   `status` (memory_status ENUM): Status ('active', 'archived', 'forgotten').
    *   `relevance_score` (REAL): Calculated relevance.

3.  **`memory_relationships`:**
    *   `id` (UUID, PRIMARY KEY): Unique identifier.
    *   `source_memory_id` (UUID): ID of the source memory.
    *   `target_memory_id` (UUID): ID of the target memory.
    *   `relationship_type` (TEXT): Type of relationship.
    *   `properties` (JSONB): Additional relationship properties.

### Memory Type Tables

Each memory type has a corresponding table:

*   **`episodic_memories`:** Stores details specific to episodic memories (see Memory Types section).
*   **`semantic_memories`:** Stores details specific to semantic memories (see Memory Types section).
*   **`procedural_memories`:** Stores details specific to procedural memories (see Memory Types section).
*   **`strategic_memories`:** Stores details specific to strategic memories (see Memory Types section).

### Indexes and Constraints

*   Vector indexes on the `embedding` column for similarity search.
*   Graph indexes for efficient relationship traversal.
*   Temporal indexes on `created_at`, `updated_at`, and `last_accessed` for time-based queries.
*   Foreign key constraints to ensure data integrity between tables.

### Triggers and Functions

*   **`track_memory_changes()`:** Trigger function to track changes to memories in the `memory_changes` table.
*   **`update_memory_timestamp()`:** Trigger function to automatically update the `updated_at` timestamp on memory updates.
*   **`update_memory_importance()`:** Trigger function to recalculate memory importance based on access count.
*   **`create_memory_relationship(source_id, target_id, relationship_type, properties)`:** Function to create relationships between memories in the graph database.
*   **`age_in_days(timestamp)`:** Function to calculate the age of a timestamp in days.

### Views
*   **`memory_health`:** A view that provides an overview of memory statistics, including total memories, average importance, and average access count for each memory type.
*   **`procedural_effectiveness`:** A view specifically for procedural memories, showing success rates, importance, and relevance scores.

### Worldview and Identity Tables

* **`worldview_primitives`**:
    * `id` (UUID, PRIMARY KEY): Unique identifier for the worldview primitive.
    * `category` (TEXT): Category of the belief (e.g., 'values', 'ethics').
    * `belief` (TEXT): Description of the belief.
    * `confidence` (REAL): Confidence level in the belief.
    * `emotional_valence` (REAL): Emotional association with the belief.
    * `stability_score` (REAL): How resistant the belief is to change.
    * `activation_patterns` (JSONB): Patterns that activate this belief.
    * `memory_filter_rules` (JSONB): Rules for filtering memories based on this belief.
    * `influence_patterns` (JSONB): How this belief influences other beliefs and memories.

* **`worldview_memory_influences`**:
    * `id` (UUID, PRIMARY KEY): Unique identifier for the influence relationship.
    * `worldview_id` (UUID): Foreign key referencing `worldview_primitives`.
    * `memory_id` (UUID): Foreign key referencing `memories`.
    * `influence_type` (TEXT): Type of influence (e.g., 'filter', 'bias').
    * `strength` (REAL): Strength of the influence.

* **`identity_model`**:
    * `id` (UUID, PRIMARY KEY): Unique identifier for the identity aspect.
    * `self_concept` (JSONB): Description of the AGI's self-concept.
    * `agency_beliefs` (JSONB): Beliefs about the AGI's own agency.
    * `purpose_framework` (JSONB): The AGI's understanding of its purpose.
    * `group_identifications` (JSONB): Groups the AGI identifies with.
    * `boundary_definitions` (JSONB): Definitions of the AGI's boundaries.
    * `emotional_baseline` (JSONB): The AGI's baseline emotional state.
    * `threat_sensitivity` (REAL): Sensitivity to perceived threats.
    * `change_resistance` (REAL): Resistance to change.

* **`identity_memory_resonance`**:
    * `id` (UUID, PRIMARY KEY): Unique identifier for the resonance relationship.
    * `memory_id` (UUID): Foreign key referencing `memories`.
    * `identity_aspect` (UUID): Foreign key referencing `identity_model`.
    * `resonance_strength` (REAL): Strength of the resonance between the memory and identity aspect.
    * `integration_status` (TEXT): Status of integration (e.g., 'integrated', 'conflicting').

## Usage Guide

### Interacting with the System

The primary way to interact with the AGI Memory System is through SQL queries and, for graph-related operations, Cypher queries via Apache AGE.  There is no separate API layer defined in this repository; all interactions are directly with the database.

### Common Operations

*   **Storing Memories:**
    *   Insert a new record into the `memories` table, specifying the `type`, `content`, and `embedding`.
    *   Insert a corresponding record into the appropriate memory type table (e.g., `episodic_memories`, `semantic_memories`) with type-specific data.

*   **Retrieving Memories:**
    *   Use `SELECT` queries on the `memories` table, filtering by `id`, `type`, `content`, or other attributes.
    *   Use vector similarity search (`<->` operator) to find memories similar to a given embedding.
    *   Use `JOIN` operations to retrieve data from both the `memories` table and the corresponding memory type table.

*   **Updating Memories:**
    *   Use `UPDATE` queries on the `memories` table to modify general memory attributes.
    *   Use `UPDATE` queries on the specific memory type tables to modify type-specific data.
    *   The `update_memory_timestamp` trigger will automatically update the `updated_at` field.
    *   The `update_memory_importance` trigger will automatically recalculate importance based on access count.

*   **Creating Relationships:**
    *   Use the `create_memory_relationship` function to create relationships between memories in the graph database.  This function takes the source memory ID, target memory ID, relationship type, and optional properties as input.

*   **Querying the Graph:**
    *   Use Cypher queries via the `ag_catalog.cypher` function to traverse and query the graph of memory relationships.

### Relevance Scoring

The `relevance_score` is calculated based on the following factors:

*   **Importance:** The inherent importance of the memory.
*   **Decay Rate:** The rate at which the memory's relevance decreases over time.
*   **Access Count:** The number of times the memory has been accessed.
*   **Age:** The time since the memory was created or last updated (calculated using the `age_in_days` function).

The `update_memory_importance` trigger automatically updates the `importance` based on the `access_count`, which in turn affects the `relevance_score`.

## System Maintenance

The memory system requires regular maintenance to ensure optimal performance and prevent excessive storage usage.

### 1. Memory Consolidation

*   **Purpose:** Move frequently accessed or important memories from `working_memory` to long-term storage (`memories` and the corresponding type-specific tables).
*   **Frequency:** Recommended every 4-6 hours.
*   **Considerations:** Memory importance, access patterns, and expiry times.

### 2. Memory Pruning

*   **Purpose:** Archive or remove low-relevance memories to free up storage space.
*   **Frequency:** Recommended daily or weekly, depending on system usage.
*   **Considerations:** Relevance scores, decay rates, and last access times.

### 3. Database Optimization

*   **Purpose:** Ensure efficient query performance.
*   **Tasks:**
    *   Reindex tables (especially vector indexes).
    *   Update database statistics.
*   **Frequency:** Recommended during off-peak hours.

These maintenance tasks can be implemented using database scheduling tools, external task schedulers, or system-level scheduling (e.g., cron, systemd).

## Example Queries

### SQL (PostgreSQL)

```sql
-- Find memories similar to a given embedding
SELECT id, content, embedding <-> '[0.1, 0.2, ...]' AS distance
FROM memories
ORDER BY distance
LIMIT 10;

-- Retrieve an episodic memory and its details
SELECT m.*, e.*
FROM memories m
JOIN episodic_memories e ON m.id = e.memory_id
WHERE m.id = 'your-memory-id';

-- Update the access count of a memory (triggers importance update)
UPDATE memories
SET access_count = access_count + 1
WHERE id = 'your-memory-id';

-- Find memories with a specific status
SELECT *
FROM memories
WHERE status = 'active';
```

### Cypher (Apache AGE)

```cypher
-- Create a relationship between two memories
SELECT create_memory_relationship(
    'source-memory-id',
    'target-memory-id',
    'RELATES_TO',
    '{"weight": 0.8}'
);

-- Find memories related to a given memory
MATCH (m:MemoryNode)-[r:RELATES_TO]->(related)
WHERE m.memory_id = 'your-memory-id'
RETURN related;

-- Find a path between two memories of specific types
MATCH p = (s:MemoryNode)-[*]->(t:MemoryNode)
WHERE s.type = 'episodic' AND t.type = 'procedural'
RETURN p;

-- Create a node
SELECT * FROM cypher('memory_graph', $$
    CREATE (n:MemoryNode {
        memory_id: 'uuid',
        type: 'semantic'
    })
    RETURN n
$$) as (result agtype);
```

## Multi-AGI Considerations

This database schema is designed for a **single** AGI instance. Supporting multiple AGI instances would require significant schema modifications, including:

*   **AGI Instance Identification:** Adding an AGI instance ID to all memory tables to distinguish between different AGIs' memories.
*   **Partitioning Strategies:** Implementing partitioning strategies for memory isolation and efficient querying within each AGI's memory space.
*   **Modified Relationship Handling:** Adapting relationship handling to allow for both intra-AGI and (potentially) inter-AGI memory sharing.
*   **Separate Working Memory:** Providing separate working memory spaces for each AGI instance.
*   **Access Controls:** Implementing access controls and memory ownership to ensure data privacy and security between AGIs.

## Conclusion

The AGI Memory System provides a comprehensive and flexible foundation for building AGI applications that require sophisticated memory management capabilities. Its multiple memory types, graph relationships, dynamic scoring, and other features enable the creation of AGIs that can effectively store, retrieve, and reason about information in a way that is inspired by human cognitive architecture. The system is designed for extensibility and can be adapted to meet the specific needs of various AGI applications.