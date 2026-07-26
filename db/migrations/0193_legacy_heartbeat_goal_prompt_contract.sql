-- Align the legacy heartbeat prompt row with the goal_priority lifecycle enum.
SET search_path = public, ag_catalog, "$user";

UPDATE prompt_modules
SET content = replace(
        content,
        '- goal_changes: Any goal priority changes you want to make',
        '- goal_changes: Any goal lifecycle changes you want to make. Use only these priority values: active, queued, backburner, completed, or abandoned.'
    ),
    updated_at = CURRENT_TIMESTAMP
WHERE key = 'heartbeat_system'
  AND content LIKE '%- goal_changes: Any goal priority changes you want to make%';
