-- Web inbox: the dashboard as a delivery endpoint of the async messaging
-- abstraction. The outbox/inbox pair (RabbitMQ hexis.outbox / hexis.inbox)
-- is a transport that arbitrary external systems hook into — email, chat
-- platforms, webhooks. The web UI is one such system: the channel worker's
-- outbox consumer tees every user-bound message it takes off the queue into
-- this table (channels/outbox.py), and the browser polls it over HTTP — a
-- pull-based endpoint, because that is what browsers speak. Read state lives
-- here (not per-browser) so every window agrees on what has been seen.
SET search_path = public, ag_catalog, "$user";
SET check_function_bodies = off;

CREATE TABLE IF NOT EXISTS web_inbox (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- The envelope message id off the queue; redelivery is a no-op.
    outbox_msg_id TEXT UNIQUE,
    kind TEXT,
    intent TEXT,
    message TEXT NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- clock_timestamp: deliveries in one batch/transaction keep their order.
    delivered_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    read_at TIMESTAMPTZ
);
ALTER TABLE web_inbox ALTER COLUMN delivered_at SET DEFAULT clock_timestamp();
CREATE INDEX IF NOT EXISTS idx_web_inbox_feed ON web_inbox (delivered_at DESC);
CREATE INDEX IF NOT EXISTS idx_web_inbox_unread ON web_inbox (delivered_at)
    WHERE read_at IS NULL;

INSERT INTO config_defaults (key, value, description) VALUES
    ('channel.web_inbox.enabled', 'true'::jsonb,
     'Deliver a copy of every user-bound outbox message to the web dashboard inbox')
ON CONFLICT (key) DO NOTHING;

-- Deliver one queue body ({id, kind, payload}) to the web endpoint.
-- Idempotent by envelope id; returns the row id, or NULL when the body
-- carries no user-readable text (nothing to show).
CREATE OR REPLACE FUNCTION web_inbox_deliver(p_body JSONB)
RETURNS UUID AS $$
DECLARE
    body JSONB := COALESCE(p_body, '{}'::jsonb);
    payload JSONB := CASE WHEN jsonb_typeof(body->'payload') = 'object'
                          THEN body->'payload' ELSE '{}'::jsonb END;
    msg TEXT;
    new_id UUID;
BEGIN
    msg := NULLIF(btrim(COALESCE(
        payload->>'message',
        payload->>'content',
        CASE WHEN jsonb_typeof(body->'payload') = 'string' THEN body->>'payload' END,
        '')), '');
    IF msg IS NULL THEN
        RETURN NULL;
    END IF;

    INSERT INTO web_inbox (outbox_msg_id, kind, intent, message, payload)
    VALUES (
        NULLIF(btrim(COALESCE(body->>'id', '')), ''),
        NULLIF(btrim(COALESCE(body->>'kind', '')), ''),
        NULLIF(btrim(COALESCE(payload->>'intent', '')), ''),
        msg,
        payload
    )
    ON CONFLICT (outbox_msg_id) DO NOTHING
    RETURNING id INTO new_id;
    RETURN new_id;
END;
$$ LANGUAGE plpgsql;

-- Queue a user-facing note whose delivery endpoint is the dashboard inbox,
-- then deliver it immediately. This is the chat/UI path: the user is already
-- looking at the dashboard, so a dashboard-only note should not depend on the
-- RabbitMQ/channel-worker relay being alive before it appears.
CREATE OR REPLACE FUNCTION queue_web_inbox_message(
    p_message TEXT,
    p_intent TEXT DEFAULT NULL,
    p_source TEXT DEFAULT 'tool'
) RETURNS JSONB AS $$
DECLARE
    outbox_id UUID;
    envelope JSONB;
    web_inbox_id UUID;
BEGIN
    outbox_id := queue_outbox_message(
        p_message,
        p_intent,
        p_source,
        '{"mode":"web_inbox"}'::jsonb
    );

    SELECT o.envelope
    INTO envelope
    FROM outbox_messages o
    WHERE o.id = outbox_id;

    web_inbox_id := web_inbox_deliver(jsonb_build_object(
        'id', envelope->>'message_id',
        'kind', envelope->>'kind',
        'payload', envelope->'payload'
    ));

    UPDATE outbox_messages
    SET status = 'published',
        claimed_at = COALESCE(claimed_at, CURRENT_TIMESTAMP),
        published_at = CURRENT_TIMESTAMP
    WHERE id = outbox_id;

    RETURN jsonb_build_object(
        'queued', true,
        'delivered', web_inbox_id IS NOT NULL,
        'outbox_id', outbox_id::text,
        'web_inbox_id', CASE WHEN web_inbox_id IS NULL THEN NULL ELSE web_inbox_id::text END,
        'delivery', jsonb_build_object('mode', 'web_inbox'),
        'envelope', envelope
    );
END;
$$ LANGUAGE plpgsql;

-- The feed the dashboard polls: unread count + newest-first messages.
CREATE OR REPLACE FUNCTION get_web_inbox(p_limit INT DEFAULT 30)
RETURNS JSONB AS $$
    SELECT jsonb_build_object(
        'unread', (SELECT COUNT(*) FROM web_inbox WHERE read_at IS NULL),
        'messages', COALESCE((
            SELECT jsonb_agg(jsonb_build_object(
                'id', m.id,
                'kind', m.kind,
                'intent', m.intent,
                'message', m.message,
                'payload', m.payload,
                'delivered_at', m.delivered_at,
                'read_at', m.read_at
            ) ORDER BY m.delivered_at DESC)
            FROM (
                SELECT * FROM web_inbox
                ORDER BY delivered_at DESC
                LIMIT GREATEST(COALESCE(p_limit, 30), 1)
            ) m
        ), '[]'::jsonb)
    );
$$ LANGUAGE sql STABLE;

CREATE OR REPLACE FUNCTION mark_web_inbox_read(p_ids UUID[])
RETURNS INT AS $$
DECLARE
    updated INT;
BEGIN
    UPDATE web_inbox
    SET read_at = CURRENT_TIMESTAMP
    WHERE id = ANY(COALESCE(p_ids, ARRAY[]::uuid[])) AND read_at IS NULL;
    GET DIAGNOSTICS updated = ROW_COUNT;
    RETURN updated;
END;
$$ LANGUAGE plpgsql;

-- Explicit dismissal from the dashboard: the user handled the message and
-- wants it gone (distinct from read, which keeps it in the feed).
CREATE OR REPLACE FUNCTION delete_web_inbox(p_ids UUID[])
RETURNS INT AS $$
DECLARE
    removed INT;
BEGIN
    DELETE FROM web_inbox
    WHERE id = ANY(COALESCE(p_ids, ARRAY[]::uuid[]));
    GET DIAGNOSTICS removed = ROW_COUNT;
    RETURN removed;
END;
$$ LANGUAGE plpgsql;
