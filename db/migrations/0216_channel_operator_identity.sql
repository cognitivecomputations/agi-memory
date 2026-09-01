-- Let every channel wizard persist an explicit operator identity. Conversation
-- allowlists remain separate and never imply policy authority.
SET search_path = public, ag_catalog, "$user";

CREATE OR REPLACE FUNCTION channel_setting_names(
    p_channel TEXT
) RETURNS TEXT[] AS $$
DECLARE
    catalog JSONB := '{
        "discord":  ["bot_token", "operator_user_id", "allowed_guilds"],
        "telegram": ["bot_token", "operator_user_id", "allowed_chat_ids"],
        "slack":    ["bot_token", "app_token", "signing_secret", "operator_user_id", "allowed_channels"],
        "signal":   ["phone_number", "api_url", "operator_user_id", "allowed_numbers"],
        "whatsapp": ["access_token", "phone_number_id", "verify_token", "webhook_port", "operator_user_id", "allowed_numbers"],
        "imessage": ["api_url", "password", "operator_recipient", "allowed_handles"],
        "matrix":   ["homeserver", "user_id", "access_token", "operator_user_id", "allowed_rooms"]
    }'::jsonb;
BEGIN
    IF NOT catalog ? COALESCE(p_channel, '') THEN
        RAISE EXCEPTION 'Unknown channel type: %; expected one of %',
            COALESCE(p_channel, '(null)'),
            (SELECT string_agg(key, ', ' ORDER BY key) FROM jsonb_object_keys(catalog) key);
    END IF;
    RETURN ARRAY(SELECT jsonb_array_elements_text(catalog->p_channel));
END;
$$ LANGUAGE plpgsql IMMUTABLE;

UPDATE integration_connectors
SET setup_manifest = jsonb_set(
        setup_manifest,
        '{config_keys}',
        COALESCE(setup_manifest->'config_keys', '[]'::jsonb)
            || jsonb_build_array('channel.telegram.operator_user_id'),
        TRUE
    ),
    updated_at = CURRENT_TIMESTAMP
WHERE id = 'telegram'
  AND NOT COALESCE(setup_manifest->'config_keys', '[]'::jsonb)
      ? 'channel.telegram.operator_user_id';

UPDATE integration_connectors
SET setup_manifest = jsonb_set(
        setup_manifest,
        '{config_keys}',
        COALESCE(setup_manifest->'config_keys', '[]'::jsonb)
            || jsonb_build_array('channel.signal.operator_user_id'),
        TRUE
    ),
    updated_at = CURRENT_TIMESTAMP
WHERE id = 'signal'
  AND NOT COALESCE(setup_manifest->'config_keys', '[]'::jsonb)
      ? 'channel.signal.operator_user_id';
