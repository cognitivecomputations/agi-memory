-- 0184: autonomous Gmail heartbeat checks require explicit Hexis authorization.

INSERT INTO config_defaults (key, value, description) VALUES
    (
        'integrations.gmail.heartbeat_digest_enabled',
        'false'::jsonb,
        'Controls whether heartbeat may proactively check connected Gmail for digests or important messages without a live user turn.'
    )
ON CONFLICT (key) DO UPDATE SET
    value = EXCLUDED.value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;
