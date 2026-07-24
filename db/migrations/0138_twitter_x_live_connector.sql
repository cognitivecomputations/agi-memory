-- Twitter/X live OAuth, read/search ingestion, and effect adapter manifest.
-- Archive import remains available as a separate local-history path.
SET search_path = public, ag_catalog, "$user";

UPDATE integration_connectors
SET auth_type = 'oauth2',
    status = 'available',
    capability_manifest = '{
      "read": {
        "label": "Read posts, mentions, and basic account identity",
        "scope_kind": "read",
        "status": "available",
        "scopes": ["tweet.read", "users.read", "offline.access"]
      },
      "search": {
        "label": "Search recent posts",
        "scope_kind": "read",
        "status": "available",
        "scopes": ["tweet.read", "users.read", "offline.access"]
      },
      "ingest": {
        "label": "Ingest live posts and mentions",
        "scope_kind": "read",
        "status": "available",
        "scopes": ["tweet.read", "users.read", "offline.access"]
      },
      "dm_read": {
        "label": "Read Direct Messages",
        "scope_kind": "read_private",
        "status": "available",
        "scopes": ["dm.read", "tweet.read", "users.read", "offline.access"]
      },
      "send": {
        "label": "Create posts and replies",
        "scope_kind": "send",
        "status": "available",
        "scopes": ["tweet.read", "tweet.write", "users.read", "offline.access"]
      },
      "dm_send": {
        "label": "Send Direct Messages",
        "scope_kind": "send_private",
        "status": "available",
        "scopes": ["dm.write", "tweet.read", "users.read", "offline.access"]
      },
      "archive_import": {
        "label": "Import historical posts and DMs from a Twitter/X archive",
        "scope_kind": "local_history",
        "status": "available",
        "scopes": []
      }
    }'::jsonb,
    setup_manifest = '{
      "flow": "oauth2_authorization_code_pkce",
      "redirect_uri": "http://localhost:1",
      "requires_user_client": true,
      "secret_storage": "~/.hexis/auth",
      "supported_surfaces": ["chat", "cli", "web", "channels"],
      "default_capabilities": ["read", "search", "ingest"],
      "capability_order": ["read", "search", "ingest", "dm_read", "send", "dm_send", "archive_import"],
      "required_scopes": [],
      "scope_order": ["tweet.read", "users.read", "offline.access", "dm.read", "tweet.write", "dm.write"],
      "history_import": {
        "flow": "local_archive_import",
        "capability": "archive_import",
        "accepted_inputs": ["export_path", "import_path"],
        "notes": ["Download your Twitter/X archive and point Hexis at the extracted archive directory or tweet/direct-message JS file."]
      },
      "capability_aliases": {
        "x": "read",
        "twitter": "read",
        "tweets": "read",
        "posts": "read",
        "timeline": "read",
        "mentions": "read",
        "search_posts": "search",
        "history": "archive_import",
        "archive": "archive_import",
        "import": "archive_import",
        "ingest_history": "archive_import",
        "dm": "dm_send",
        "dms": "dm_read",
        "direct_messages": "dm_read",
        "message": "dm_send",
        "write": "send",
        "post": "send",
        "reply": "send",
        "respond": "send"
      },
      "user_next_step": "Create or choose an X Developer app, enable user sign-in, register http://localhost:1 as the callback URI, then start Twitter/X connection setup. Request only the capabilities you want; archive import is still available through a local export path."
    }'::jsonb,
    metadata = COALESCE(metadata, '{}'::jsonb) || jsonb_build_object(
        'twitter_x_live_oauth', true,
        'twitter_x_archive_import', true,
        'official_docs', jsonb_build_array(
            'https://docs.x.com/fundamentals/authentication/oauth-2-0/authorization-code',
            'https://docs.x.com/x-api/posts/creation-of-a-post',
            'https://docs.x.com/x-api/direct-messages/quickstart',
            'https://docs.x.com/fundamentals/rate-limits',
            'https://docs.x.com/x-api/getting-started/pricing'
        ),
        'updated_by', 'db/migrations/0138_twitter_x_live_connector.sql'
    ),
    docs_url = 'https://docs.x.com',
    updated_at = CURRENT_TIMESTAMP
WHERE id = 'twitter_x';

INSERT INTO connector_action_tool_map (
    tool_name,
    connector_id,
    action_kind,
    target_argument,
    account_argument,
    sensitivity,
    metadata
) VALUES
    ('twitter_x_post', 'twitter_x', 'post', 'text', 'account_key', 'external_message',
     '{"tool_module": "core.tools.twitter_x_actions", "provider_endpoint": "POST /2/tweets", "cost_basis": "per_request"}'::jsonb),
    ('twitter_x_reply', 'twitter_x', 'reply', 'reply_to_tweet_id', 'account_key', 'external_message',
     '{"tool_module": "core.tools.twitter_x_actions", "provider_endpoint": "POST /2/tweets", "cost_basis": "per_request"}'::jsonb),
    ('twitter_x_dm_send', 'twitter_x', 'dm_send', 'participant_id', 'account_key', 'external_message',
     '{"tool_module": "core.tools.twitter_x_actions", "provider_endpoint": "POST /2/dm_conversations/with/:participant_id/messages", "cost_basis": "per_request"}'::jsonb)
ON CONFLICT (tool_name) DO UPDATE SET
    connector_id = EXCLUDED.connector_id,
    action_kind = EXCLUDED.action_kind,
    target_argument = EXCLUDED.target_argument,
    account_argument = EXCLUDED.account_argument,
    sensitivity = EXCLUDED.sensitivity,
    enabled = TRUE,
    metadata = (connector_action_tool_map.metadata - 'planned_tool') || EXCLUDED.metadata,
    updated_at = CURRENT_TIMESTAMP;

CREATE OR REPLACE FUNCTION estimate_connector_backfill(
    p_connector_id TEXT,
    p_requested_range JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    connector TEXT := lower(NULLIF(btrim(COALESCE(p_connector_id, '')), ''));
    requested JSONB := COALESCE(p_requested_range, '{}'::jsonb);
    max_messages INT;
    page_size INT;
    pages INT;
    export_path TEXT;
    twitter_stream TEXT;
BEGIN
    BEGIN
        max_messages := NULLIF(requested->>'max_messages', '')::int;
    EXCEPTION WHEN OTHERS THEN
        max_messages := NULL;
    END;
    BEGIN
        page_size := NULLIF(requested->>'page_size', '')::int;
    EXCEPTION WHEN OTHERS THEN
        page_size := NULL;
    END;
    max_messages := LEAST(GREATEST(COALESCE(max_messages, 100), 1), 5000);
    page_size := LEAST(GREATEST(COALESCE(page_size, 100), 1), 500);
    pages := CEIL(max_messages::numeric / page_size::numeric)::int;
    export_path := NULLIF(btrim(COALESCE(requested->>'export_path', requested->>'import_path', '')), '');
    twitter_stream := lower(NULLIF(btrim(COALESCE(requested->>'stream', requested->>'source', '')), ''));

    IF connector IN ('gmail', 'slack') THEN
        RETURN jsonb_build_object(
            'connector_id', connector,
            'provider_status', 'api_backfill_available',
            'estimated_items', max_messages,
            'page_size', page_size,
            'estimated_pages', pages,
            'cost_class', CASE WHEN max_messages <= 100 THEN 'small'
                               WHEN max_messages <= 1000 THEN 'medium'
                               ELSE 'large' END,
            'rate_limit_notes', CASE connector
                WHEN 'gmail' THEN 'Gmail API quota and query selectivity determine runtime.'
                ELSE 'Slack conversations.history pagination and workspace rate limits determine runtime.'
            END
        );
    ELSIF connector IN ('telegram', 'signal') THEN
        RETURN jsonb_build_object(
            'connector_id', connector,
            'provider_status', CASE WHEN export_path IS NULL THEN 'export_required' ELSE 'local_export_import' END,
            'estimated_items', max_messages,
            'page_size', page_size,
            'estimated_pages', pages,
            'cost_class', CASE WHEN export_path IS NULL THEN 'blocked_until_export'
                               WHEN max_messages <= 1000 THEN 'local_medium'
                               ELSE 'local_large' END,
            'requires_export_path', export_path IS NULL,
            'limitations', CASE connector
                WHEN 'telegram' THEN 'Telegram Bot API cannot retroactively read chat history; import a Telegram data export for history.'
                ELSE 'Signal runtime APIs do not expose retro-history; import a local Signal export/source artifact for history.'
            END
        );
    ELSIF connector = 'twitter_x' THEN
        IF export_path IS NOT NULL THEN
            RETURN jsonb_build_object(
                'connector_id', connector,
                'provider_status', 'local_archive_import',
                'estimated_items', max_messages,
                'page_size', page_size,
                'estimated_pages', pages,
                'cost_class', CASE WHEN max_messages <= 1000 THEN 'local_medium' ELSE 'local_large' END,
                'requires_export_path', FALSE,
                'pricing_notes', 'Local archive import does not call the X API.'
            );
        END IF;
        RETURN jsonb_build_object(
            'connector_id', connector,
            'provider_status', 'api_backfill_available',
            'stream', COALESCE(twitter_stream, 'timeline'),
            'estimated_items', max_messages,
            'page_size', page_size,
            'estimated_pages', pages,
            'cost_class', CASE WHEN max_messages <= 100 THEN 'paid_small'
                               WHEN max_messages <= 1000 THEN 'paid_medium'
                               ELSE 'paid_large' END,
            'pricing_notes', 'X API reads are pay-per-resource; check current X Developer Console pricing and spend limits before large imports.',
            'rate_limit_notes', 'X API v2 uses per-endpoint 15-minute windows and returns x-rate-limit-* headers.'
        );
    END IF;

    RETURN jsonb_build_object(
        'connector_id', connector,
        'provider_status', 'unknown',
        'estimated_items', max_messages,
        'estimated_pages', pages,
        'cost_class', 'unknown'
    );
END;
$$;
