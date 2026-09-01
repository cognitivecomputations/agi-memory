-- Hexis schema: triggers.
SET search_path = public, ag_catalog, "$user";
CREATE TRIGGER trg_memory_temporal_validity_insert
    BEFORE INSERT ON memories
    FOR EACH ROW
    EXECUTE FUNCTION normalize_memory_temporal_validity();
CREATE TRIGGER trg_memory_temporal_validity_update
    BEFORE UPDATE OF valid_from, valid_until, superseded_by ON memories
    FOR EACH ROW
    EXECUTE FUNCTION normalize_memory_temporal_validity();
CREATE TRIGGER trg_memory_supersession_insert
    AFTER INSERT ON memories
    FOR EACH ROW
    WHEN (NEW.superseded_by IS NOT NULL)
    EXECUTE FUNCTION sync_memory_supersession_from_memory();
CREATE TRIGGER trg_memory_supersession_update
    AFTER UPDATE OF superseded_by ON memories
    FOR EACH ROW
    WHEN (OLD.superseded_by IS DISTINCT FROM NEW.superseded_by)
    EXECUTE FUNCTION sync_memory_supersession_from_memory();
CREATE TRIGGER trg_memory_embedding_lifecycle
    BEFORE INSERT OR UPDATE OF embedding, embedding_status ON memories
    FOR EACH ROW
    EXECUTE FUNCTION normalize_memory_embedding_lifecycle();
CREATE TRIGGER trg_memory_timestamp
    BEFORE UPDATE ON memories
    FOR EACH ROW
    EXECUTE FUNCTION update_memory_timestamp();
CREATE TRIGGER trg_importance_on_access
    BEFORE UPDATE ON memories
    FOR EACH ROW
    WHEN (NEW.access_count != OLD.access_count)
    EXECUTE FUNCTION update_memory_importance();
CREATE TRIGGER trg_neighborhood_staleness
    AFTER UPDATE OF embedding, importance, status ON memories
    FOR EACH ROW
    EXECUTE FUNCTION mark_neighborhoods_stale();
CREATE TRIGGER trg_auto_episode_assignment
    AFTER INSERT ON memories
    FOR EACH ROW
    EXECUTE FUNCTION assign_to_episode();
CREATE TRIGGER trg_auto_worldview_alignment
    AFTER INSERT ON memories
    FOR EACH ROW
    EXECUTE FUNCTION auto_check_worldview_alignment();
CREATE TRIGGER trg_auto_worldview_alignment_embedding
    AFTER UPDATE OF embedding ON memories
    FOR EACH ROW
    WHEN (NEW.embedding IS NOT NULL)
    EXECUTE FUNCTION auto_check_worldview_alignment();
CREATE TRIGGER trg_heartbeat_state_update
INSTEAD OF UPDATE ON heartbeat_state
FOR EACH ROW
EXECUTE FUNCTION heartbeat_state_update_trigger();
CREATE TRIGGER trg_heartbeat_economy_state_transition
AFTER UPDATE ON state
FOR EACH ROW
WHEN (OLD.key = 'heartbeat_state' AND NEW.key = 'heartbeat_state')
EXECUTE FUNCTION heartbeat_economy_state_transition_trigger();
CREATE TRIGGER trg_maintenance_state_update
INSTEAD OF UPDATE ON maintenance_state
FOR EACH ROW
EXECUTE FUNCTION maintenance_state_update_trigger();
CREATE TRIGGER memories_emotional_context_insert
BEFORE INSERT ON memories
FOR EACH ROW
WHEN (current_setting('hexis.hmx_import', true) IS DISTINCT FROM 'on')
EXECUTE FUNCTION apply_emotional_context_to_memory();
CREATE TRIGGER trg_memory_goal_origin
BEFORE INSERT OR UPDATE OF type, goal_origin, metadata ON memories
FOR EACH ROW
EXECUTE FUNCTION normalize_memory_goal_origin();
-- HMX Slice 0: init-created memories get bootstrap provenance at creation.
-- The WHEN predicate inlines is_initialization_memory() — keep the three
-- marker checks in sync with it and with reset_persona().
CREATE TRIGGER trg_hmx_bootstrap_provenance
    BEFORE INSERT OR UPDATE ON memories
    FOR EACH ROW
    WHEN (
        NEW.metadata->'provenance' IS NULL
        AND (NEW.metadata->>'origin' = 'initialization'
             OR NEW.metadata->>'type' = 'initialization'
             OR NEW.source_attribution->>'source' = 'initialization')
    )
    EXECUTE FUNCTION hmx_tag_bootstrap_provenance();

CREATE TRIGGER trg_hmx_drive_provenance
    BEFORE UPDATE ON drives
    FOR EACH ROW
    EXECUTE FUNCTION hmx_mark_drive_experienced();

CREATE TRIGGER trg_hmx_emotional_trigger_provenance
    BEFORE INSERT ON emotional_triggers
    FOR EACH ROW
    EXECUTE FUNCTION hmx_default_emotional_trigger_provenance();

DROP TRIGGER IF EXISTS trg_relationship_injury_from_subconscious_unit ON subconscious_units;
CREATE TRIGGER trg_relationship_injury_from_subconscious_unit
    AFTER INSERT OR UPDATE OF status ON subconscious_units
    FOR EACH ROW
    EXECUTE FUNCTION relationship_injury_from_subconscious_unit();

DROP TRIGGER IF EXISTS trg_channel_message_source_artifact ON channel_messages;
CREATE TRIGGER trg_channel_message_source_artifact
    AFTER INSERT ON channel_messages
    FOR EACH ROW
    EXECUTE FUNCTION channel_message_source_artifact_trigger();

DROP TRIGGER IF EXISTS trg_tool_surface_decisions_immutable ON tool_surface_decision_events;
CREATE TRIGGER trg_tool_surface_decisions_immutable
    BEFORE UPDATE OR DELETE ON tool_surface_decision_events
    FOR EACH ROW
    EXECUTE FUNCTION reject_tool_surface_audit_mutation();

DROP TRIGGER IF EXISTS trg_voice_note_stt_events_immutable ON voice_note_stt_events;
CREATE TRIGGER trg_voice_note_stt_events_immutable
    BEFORE UPDATE OR DELETE ON voice_note_stt_events
    FOR EACH ROW
    EXECUTE FUNCTION reject_tool_surface_audit_mutation();
