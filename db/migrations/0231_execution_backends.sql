-- Explicit execution placement. Profiles contain endpoints and local file
-- references only; private keys and credentials never enter the database.
SET search_path = public, ag_catalog, "$user";

INSERT INTO config_defaults (key, value, description) VALUES
    ('execution.backends',
     '{"active":"local","profiles":{"local":{"type":"local"}}}'::jsonb,
     'Named execution profiles and the explicit active profile for shell, script, and code tools.'),
    ('execution.max_output_chars', '50000'::jsonb,
     'Maximum stdout or stderr characters returned by one execution tool call.'),
    ('execution.max_timeout_seconds', '300'::jsonb,
     'Global timeout ceiling applied by every execution backend.'),
    ('execution.repl_state_ttl_hours', '168'::jsonb,
     'Hours to retain inactive remote execute_code session state before bounded cleanup.')
ON CONFLICT (key) DO NOTHING;
