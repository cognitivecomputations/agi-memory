-- Enforce migration checksum integrity and preserve explicit contradiction edges.
--
-- Three migration files were amended during the still-unpublished implementation
-- cycle after a development database had already applied them. Record that exact,
-- bounded reconciliation before the runner begins enforcing immutable checksums.
-- Fresh databases already record the canonical hashes and insert no audit rows.
SET search_path = public, ag_catalog, "$user";

CREATE TABLE IF NOT EXISTS migration_checksum_reconciliations (
    version TEXT NOT NULL,
    previous_checksum TEXT NOT NULL,
    canonical_checksum TEXT NOT NULL,
    reason TEXT NOT NULL,
    reconciled_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (version, previous_checksum, canonical_checksum)
);

WITH reconciliations(version, previous_checksum, canonical_checksum, reason) AS (
    VALUES
        (
            '0121_source_retention',
            '9eb458a74cecd8e81ef0b23c1aa5f3d307e5825c440ae0a42a0e7da285223755',
            '1cd180b591938cc0b758f83df28bb0c72599b38ca4959f0caa5574ef5aaf0a6c',
            'Pre-publication retention-default documentation correction'
        ),
        (
            '0233_contradiction_events',
            '456abf22d10d86c18e8b0fe71458183512791474e4a57076a75a950f2d7c83fb',
            'd9fbebb0e7962e521cc6f3fae7e7053987adefbe5a3dccfeacaf30b0459246a4',
            'Pre-publication contradiction-event hardening'
        ),
        (
            '0234_temporal_memory_history',
            '415da496a59a51f1ec2d2d73ca4f1b3fab6938e2da49682591a9bbcb5079e9fa',
            '7b27a667944e4bbcbce0393b70483460e0d45add84679958e7a72760d6210981',
            'Pre-publication temporal-history hardening'
        )
)
INSERT INTO migration_checksum_reconciliations (
    version, previous_checksum, canonical_checksum, reason
)
SELECT r.version, r.previous_checksum, r.canonical_checksum, r.reason
FROM reconciliations r
JOIN public.schema_migrations m
  ON m.version = r.version
 AND m.checksum = r.previous_checksum
ON CONFLICT DO NOTHING;

WITH reconciliations(version, previous_checksum, canonical_checksum) AS (
    VALUES
        (
            '0121_source_retention',
            '9eb458a74cecd8e81ef0b23c1aa5f3d307e5825c440ae0a42a0e7da285223755',
            '1cd180b591938cc0b758f83df28bb0c72599b38ca4959f0caa5574ef5aaf0a6c'
        ),
        (
            '0233_contradiction_events',
            '456abf22d10d86c18e8b0fe71458183512791474e4a57076a75a950f2d7c83fb',
            'd9fbebb0e7962e521cc6f3fae7e7053987adefbe5a3dccfeacaf30b0459246a4'
        ),
        (
            '0234_temporal_memory_history',
            '415da496a59a51f1ec2d2d73ca4f1b3fab6938e2da49682591a9bbcb5079e9fa',
            '7b27a667944e4bbcbce0393b70483460e0d45add84679958e7a72760d6210981'
        )
)
UPDATE public.schema_migrations m
SET checksum = r.canonical_checksum
FROM reconciliations r
WHERE m.version = r.version
  AND m.checksum = r.previous_checksum;

-- Contradiction cases are authoritative for detected/reviewable conflicts, but
-- connect_memories(..., CONTRADICTS) has long promised that an explicit graph
-- relationship is returned by find_contradictions(). Retain that compatibility
-- without duplicating pairs already represented by a pending ledger case.
CREATE OR REPLACE FUNCTION find_contradictions(p_memory_id UUID DEFAULT NULL)
RETURNS TABLE (
    memory_a UUID,
    memory_b UUID,
    content_a TEXT,
    content_b TEXT
)
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    v_filter TEXT;
BEGIN
    RETURN QUERY
    SELECT c.memory_a, c.memory_b, a.content, b.content
    FROM contradiction_cases c
    JOIN memories a ON a.id = c.memory_a
    JOIN memories b ON b.id = c.memory_b
    WHERE c.status = 'pending'
      AND (p_memory_id IS NULL OR p_memory_id IN (c.memory_a, c.memory_b))
    ORDER BY c.confidence DESC, c.detected_at, c.id;

    v_filter := CASE
        WHEN p_memory_id IS NULL THEN ''
        ELSE format(
            'WHERE a.memory_id = %L OR b.memory_id = %L',
            p_memory_id,
            p_memory_id
        )
    END;
    BEGIN
        RETURN QUERY EXECUTE format($query$
            WITH graph_pairs AS (
                SELECT DISTINCT
                    LEAST(
                        replace(a_id::text, '"', '')::uuid,
                        replace(b_id::text, '"', '')::uuid
                    ) AS a_uuid,
                    GREATEST(
                        replace(a_id::text, '"', '')::uuid,
                        replace(b_id::text, '"', '')::uuid
                    ) AS b_uuid
                FROM ag_catalog.cypher('memory_graph', $cypher$
                    MATCH (a:MemoryNode)-[:CONTRADICTS]-(b:MemoryNode)
                    %s
                    RETURN a.memory_id, b.memory_id
                $cypher$) AS (a_id ag_catalog.agtype, b_id ag_catalog.agtype)
            )
            SELECT p.a_uuid, p.b_uuid, a.content, b.content
            FROM graph_pairs p
            JOIN memories a ON a.id = p.a_uuid
            JOIN memories b ON b.id = p.b_uuid
            WHERE NOT EXISTS (
                SELECT 1
                FROM contradiction_cases c
                WHERE c.status = 'pending'
                  AND LEAST(c.memory_a, c.memory_b) = p.a_uuid
                  AND GREATEST(c.memory_a, c.memory_b) = p.b_uuid
            )
        $query$, v_filter);
    EXCEPTION WHEN OTHERS THEN
        RAISE DEBUG 'Legacy contradiction graph lookup unavailable: %', SQLERRM;
    END;
END;
$$;
