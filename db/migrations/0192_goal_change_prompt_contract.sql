-- Align DB-owned heartbeat prompt modules with the real goal_priority contract.
SET search_path = public, ag_catalog, "$user";

UPDATE prompt_modules
SET content = replace(
        content,
        '- **goal_changes**: Any goal priority changes (list of objects with `goal_id`, `new_priority`, `reason`)',
        '- **goal_changes**: Any goal lifecycle changes (list of objects with `goal_id`, `new_priority`, `reason`). Use only these `new_priority` values: `active`, `queued`, `backburner`, `completed`, or `abandoned`. Do not use urgency labels like `high`, `medium`, or `low`.'
    ),
    updated_at = CURRENT_TIMESTAMP
WHERE key IN ('heartbeat_system', 'rlm_heartbeat_system')
  AND content LIKE '%Any goal priority changes (list of objects with `goal_id`, `new_priority`, `reason`)%';
