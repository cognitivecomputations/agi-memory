-- Required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;
CREATE EXTENSION IF NOT EXISTS btree_gist;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS cube;

-- Load AGE extension explicitly
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

-- Create the graph
SELECT create_graph('memory_graph');
SELECT create_vlabel('memory_graph', 'MemoryNode');

-- Switch to public schema for our tables
SET search_path = public, ag_catalog, "$user";

-- Enums for memory types and status
CREATE TYPE memory_type AS ENUM ('episodic', 'semantic', 'procedural', 'strategic');
CREATE TYPE memory_status AS ENUM ('active', 'archived', 'invalidated');
CREATE TYPE cluster_type AS ENUM ('theme', 'emotion', 'temporal', 'person', 'pattern', 'mixed');

-- Working Memory (temporary table or in-memory structure)
CREATE TABLE working_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    content TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    expiry TIMESTAMPTZ
);

CREATE OR REPLACE FUNCTION age_in_days(created_at TIMESTAMPTZ) 
RETURNS FLOAT
IMMUTABLE
AS $$
BEGIN
    RETURN extract(epoch from (now() - created_at))/86400.0;
END;
$$ LANGUAGE plpgsql;

-- Base memory table with vector embeddings
CREATE TABLE memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    type memory_type NOT NULL,
    status memory_status DEFAULT 'active',
    content TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    importance FLOAT DEFAULT 0.0,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ,
    decay_rate FLOAT DEFAULT 0.01,
    relevance_score FLOAT GENERATED ALWAYS AS (
        importance * exp(-decay_rate * age_in_days(created_at))
    ) STORED
);

-- Memory clusters for thematic grouping
CREATE TABLE memory_clusters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    cluster_type cluster_type NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    centroid_embedding vector(1536), -- Average embedding of all memories in cluster
    emotional_signature JSONB, -- Common emotional patterns
    keywords TEXT[], -- Key terms associated with this cluster
    importance_score FLOAT DEFAULT 0.0,
    coherence_score FLOAT, -- How tightly related are the memories
    last_activated TIMESTAMPTZ,
    activation_count INTEGER DEFAULT 0,
    worldview_alignment FLOAT -- How much this cluster aligns with current worldview
);

-- Mapping between memories and clusters (many-to-many)
CREATE TABLE memory_cluster_members (
    cluster_id UUID REFERENCES memory_clusters(id) ON DELETE CASCADE,
    memory_id UUID REFERENCES memories(id) ON DELETE CASCADE,
    membership_strength FLOAT DEFAULT 1.0, -- How strongly this memory belongs to cluster
    added_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    contribution_to_centroid FLOAT, -- How much this memory shapes the cluster
    PRIMARY KEY (cluster_id, memory_id)
);

-- Relationships between clusters
CREATE TABLE cluster_relationships (
    from_cluster_id UUID REFERENCES memory_clusters(id) ON DELETE CASCADE,
    to_cluster_id UUID REFERENCES memory_clusters(id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL, -- 'causes', 'contradicts', 'supports', 'evolves_into'
    strength FLOAT DEFAULT 0.5,
    discovered_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    evidence_memories UUID[], -- Memory IDs that support this relationship
    PRIMARY KEY (from_cluster_id, to_cluster_id, relationship_type)
);

-- Cluster activation patterns
CREATE TABLE cluster_activation_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cluster_id UUID REFERENCES memory_clusters(id),
    activated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    activation_context TEXT, -- What triggered this activation
    activation_strength FLOAT,
    co_activated_clusters UUID[], -- Other clusters activated at same time
    resulting_insights JSONB -- Any new connections discovered
);

-- Episodic memories
CREATE TABLE episodic_memories (
    memory_id UUID PRIMARY KEY REFERENCES memories(id),
    action_taken JSONB,
    context JSONB,
    result JSONB,
    emotional_valence FLOAT,
    verification_status BOOLEAN,
    event_time TIMESTAMPTZ,
    CONSTRAINT valid_emotion CHECK (emotional_valence >= -1 AND emotional_valence <= 1)
);

-- Semantic memories
CREATE TABLE semantic_memories (
    memory_id UUID PRIMARY KEY REFERENCES memories(id),
    confidence FLOAT NOT NULL,
    last_validated TIMESTAMPTZ,
    source_references JSONB,
    contradictions JSONB,
    category TEXT[],
    related_concepts TEXT[],
    CONSTRAINT valid_confidence CHECK (confidence >= 0 AND confidence <= 1)
);

-- Procedural memories
CREATE TABLE procedural_memories (
    memory_id UUID PRIMARY KEY REFERENCES memories(id),
    steps JSONB NOT NULL,
    prerequisites JSONB,
    success_count INTEGER DEFAULT 0,
    total_attempts INTEGER DEFAULT 0,
    success_rate FLOAT GENERATED ALWAYS AS (
        CASE WHEN total_attempts > 0 
        THEN success_count::FLOAT / total_attempts::FLOAT 
        ELSE 0 END
    ) STORED,
    average_duration INTERVAL,
    failure_points JSONB
);

-- Strategic memories
CREATE TABLE strategic_memories (
    memory_id UUID PRIMARY KEY REFERENCES memories(id),
    pattern_description TEXT NOT NULL,
    supporting_evidence JSONB,
    confidence_score FLOAT,
    success_metrics JSONB,
    adaptation_history JSONB,
    context_applicability JSONB,
    CONSTRAINT valid_confidence CHECK (confidence_score >= 0 AND confidence_score <= 1)
);

-- Worldview primitives
CREATE TABLE worldview_primitives (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    category TEXT NOT NULL,
    belief TEXT NOT NULL,
    confidence FLOAT,
    emotional_valence FLOAT,
    stability_score FLOAT,
    connected_beliefs UUID[],
    activation_patterns JSONB,
    memory_filter_rules JSONB,
    influence_patterns JSONB,
    preferred_clusters UUID[] -- Clusters that align with this worldview
);

-- Track how worldview affects memory interpretation
CREATE TABLE worldview_memory_influences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    worldview_id UUID REFERENCES worldview_primitives(id),
    memory_id UUID REFERENCES memories(id),
    influence_type TEXT,
    strength FLOAT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Identity model
CREATE TABLE identity_model (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    self_concept JSONB,
    agency_beliefs JSONB,
    purpose_framework JSONB,
    group_identifications JSONB,
    boundary_definitions JSONB,
    emotional_baseline JSONB,
    threat_sensitivity FLOAT,
    change_resistance FLOAT,
    core_memory_clusters UUID[] -- Clusters central to identity
);

-- Bridge between memories and identity
CREATE TABLE identity_memory_resonance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id UUID REFERENCES memories(id),
    identity_aspect UUID REFERENCES identity_model(id),
    resonance_strength FLOAT,
    integration_status TEXT
);

-- Temporal tracking
CREATE TABLE memory_changes (
    change_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id UUID REFERENCES memories(id),
    changed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    change_type TEXT NOT NULL,
    old_value JSONB,
    new_value JSONB
);

-- Indexes for performance
CREATE INDEX ON memories USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX ON memories (status);
CREATE INDEX ON memories USING GIN (content gin_trgm_ops);
CREATE INDEX ON memories (relevance_score DESC) WHERE status = 'active';
CREATE INDEX ON memory_clusters USING ivfflat (centroid_embedding vector_cosine_ops);
CREATE INDEX ON memory_clusters (cluster_type, importance_score DESC);
CREATE INDEX ON memory_clusters (last_activated DESC);
CREATE INDEX ON memory_cluster_members (memory_id);
CREATE INDEX ON memory_cluster_members (cluster_id, membership_strength DESC);
CREATE INDEX ON cluster_relationships (from_cluster_id);
CREATE INDEX ON cluster_relationships (to_cluster_id);
CREATE INDEX ON worldview_memory_influences (memory_id, strength DESC);
CREATE INDEX ON identity_memory_resonance (memory_id, resonance_strength DESC);

-- Functions for memory management

-- Update memory timestamp
CREATE OR REPLACE FUNCTION update_memory_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Update memory importance based on access
CREATE OR REPLACE FUNCTION update_memory_importance()
RETURNS TRIGGER AS $$
BEGIN
    NEW.importance = NEW.importance * (1.0 + (ln(NEW.access_count + 1) * 0.1));
    NEW.last_accessed = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Update cluster when accessed
CREATE OR REPLACE FUNCTION update_cluster_activation()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_activated = CURRENT_TIMESTAMP;
    NEW.activation_count = NEW.activation_count + 1;
    NEW.importance_score = NEW.importance_score * (1.0 + (ln(NEW.activation_count + 1) * 0.05));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function to recalculate cluster centroid
CREATE OR REPLACE FUNCTION recalculate_cluster_centroid(cluster_uuid UUID)
RETURNS VOID AS $$
DECLARE
    new_centroid vector(1536);
BEGIN
    -- Calculate average embedding of all active memories in cluster
    SELECT AVG(m.embedding)::vector(1536)
    INTO new_centroid
    FROM memories m
    JOIN memory_cluster_members mcm ON m.id = mcm.memory_id
    WHERE mcm.cluster_id = cluster_uuid
    AND m.status = 'active'
    AND mcm.membership_strength > 0.3;
    
    UPDATE memory_clusters
    SET centroid_embedding = new_centroid,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = cluster_uuid;
END;
$$ LANGUAGE plpgsql;

-- Function to find or create cluster for a memory
CREATE OR REPLACE FUNCTION assign_memory_to_clusters(memory_uuid UUID, max_clusters INT DEFAULT 3)
RETURNS VOID AS $$
DECLARE
    memory_embedding vector(1536);
    memory_content TEXT;
    cluster_record RECORD;
    similarity_threshold FLOAT := 0.7;
    assigned_count INT := 0;
BEGIN
    -- Get memory details
    SELECT embedding, content INTO memory_embedding, memory_content
    FROM memories WHERE id = memory_uuid;
    
    -- Find similar clusters
    FOR cluster_record IN 
        SELECT id, 1 - (centroid_embedding <=> memory_embedding) as similarity
        FROM memory_clusters
        WHERE status = 'active'
        ORDER BY centroid_embedding <=> memory_embedding
        LIMIT 10
    LOOP
        IF cluster_record.similarity >= similarity_threshold AND assigned_count < max_clusters THEN
            -- Add to cluster
            INSERT INTO memory_cluster_members (cluster_id, memory_id, membership_strength)
            VALUES (cluster_record.id, memory_uuid, cluster_record.similarity)
            ON CONFLICT DO NOTHING;
            
            assigned_count := assigned_count + 1;
        END IF;
    END LOOP;
    
    -- If no suitable clusters found, consider creating a new one
    -- (This would be triggered by application logic based on themes)
END;
$$ LANGUAGE plpgsql;

-- Create memory relationship in graph
CREATE OR REPLACE FUNCTION create_memory_relationship(
    from_id UUID,
    to_id UUID,
    relationship_type TEXT,
    properties JSONB DEFAULT '{}'
) RETURNS VOID AS $$
BEGIN
    EXECUTE format(
        'SELECT * FROM cypher(''memory_graph'', $q$
            MATCH (a:MemoryNode), (b:MemoryNode)
            WHERE a.memory_id = %L AND b.memory_id = %L
            CREATE (a)-[r:%s %s]->(b)
            RETURN r
        $q$) as (result agtype)',
        from_id,
        to_id,
        relationship_type,
        case when properties = '{}'::jsonb 
             then '' 
             else format('{%s}', 
                  (SELECT string_agg(format('%I: %s', key, value), ', ')
                   FROM jsonb_each(properties)))
        end
    );
END;
$$ LANGUAGE plpgsql;

-- Triggers
CREATE TRIGGER update_memory_timestamp
    BEFORE UPDATE ON memories
    FOR EACH ROW
    EXECUTE FUNCTION update_memory_timestamp();

CREATE TRIGGER update_importance_on_access
    BEFORE UPDATE ON memories
    FOR EACH ROW
    WHEN (NEW.access_count != OLD.access_count)
    EXECUTE FUNCTION update_memory_importance();

CREATE TRIGGER update_cluster_on_access
    BEFORE UPDATE ON memory_clusters
    FOR EACH ROW
    WHEN (NEW.activation_count != OLD.activation_count)
    EXECUTE FUNCTION update_cluster_activation();

-- Views for memory analysis

CREATE VIEW memory_health AS
SELECT 
    type,
    count(*) as total_memories,
    avg(importance) as avg_importance,
    avg(access_count) as avg_access_count,
    count(*) FILTER (WHERE last_accessed > CURRENT_TIMESTAMP - INTERVAL '1 day') as accessed_last_day,
    avg(relevance_score) as avg_relevance
FROM memories
GROUP BY type;

CREATE VIEW cluster_insights AS
SELECT 
    mc.id,
    mc.name,
    mc.cluster_type,
    mc.importance_score,
    mc.coherence_score,
    count(mcm.memory_id) as memory_count,
    mc.last_activated,
    array_agg(DISTINCT cr.to_cluster_id) as related_clusters
FROM memory_clusters mc
LEFT JOIN memory_cluster_members mcm ON mc.id = mcm.cluster_id
LEFT JOIN cluster_relationships cr ON mc.id = cr.from_cluster_id
GROUP BY mc.id, mc.name, mc.cluster_type, mc.importance_score, mc.coherence_score, mc.last_activated
ORDER BY mc.importance_score DESC;

CREATE VIEW active_themes AS
SELECT 
    mc.name as theme,
    mc.emotional_signature,
    mc.keywords,
    count(DISTINCT mch.id) as recent_activations,
    array_agg(DISTINCT mch.co_activated_clusters) as associated_themes
FROM memory_clusters mc
JOIN cluster_activation_history mch ON mc.id = mch.cluster_id
WHERE mch.activated_at > CURRENT_TIMESTAMP - INTERVAL '7 days'
GROUP BY mc.id, mc.name, mc.emotional_signature, mc.keywords
ORDER BY count(DISTINCT mch.id) DESC;
