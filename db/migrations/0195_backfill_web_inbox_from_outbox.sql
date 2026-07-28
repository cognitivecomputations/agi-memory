-- 0195: Backfill the web inbox from already-"published" outbox messages (#107).
--
-- Every outbox message was marked 'published' as soon as it reached RabbitMQ,
-- but no consumer ever delivered them to a user-visible surface, so the
-- dashboard inbox stayed empty while user-bound messages accumulated.
-- Re-deliver them to web_inbox here. Idempotent: web_inbox dedupes on
-- outbox_msg_id (ON CONFLICT DO NOTHING inside web_inbox_deliver), so rows
-- already delivered — or delivered later by the channel worker — are no-ops.
DO $$
DECLARE
    r RECORD;
    delivered INT := 0;
BEGIN
    IF to_regclass('public.outbox_messages') IS NULL
       OR to_regprocedure('web_inbox_deliver(jsonb)') IS NULL THEN
        RETURN;
    END IF;

    FOR r IN
        SELECT envelope
        FROM outbox_messages
        WHERE status = 'published'
          AND COALESCE(envelope->'payload'->'delivery'->>'mode', '') <> 'silent'
        ORDER BY created_at
    LOOP
        BEGIN
            IF web_inbox_deliver(jsonb_build_object(
                'id', r.envelope->>'message_id',
                'kind', r.envelope->>'kind',
                'payload', r.envelope->'payload'
            )) IS NOT NULL THEN
                delivered := delivered + 1;
            END IF;
        EXCEPTION WHEN OTHERS THEN
            -- A malformed envelope must not block the rest of the backfill.
            NULL;
        END;
    END LOOP;

    RAISE NOTICE 'web_inbox backfill delivered % message(s)', delivered;
END $$;
