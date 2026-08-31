-- Coalesce periodic PWA presence beacons instead of retaining an unbounded log.
SET search_path = public, ag_catalog, "$user";

CREATE OR REPLACE FUNCTION record_pwa_presence(
    p_device_id TEXT,
    p_presence_kind TEXT,
    p_display_mode TEXT DEFAULT NULL,
    p_visibility TEXT DEFAULT NULL
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_device_id TEXT := NULLIF(left(btrim(COALESCE(p_device_id, '')), 200), '');
    v_event JSONB;
BEGIN
    IF v_device_id IS NULL THEN
        RAISE EXCEPTION 'PWA device_id is required';
    END IF;

    PERFORM pg_advisory_xact_lock(hashtextextended('pwa-presence:' || v_device_id, 0));
    DELETE FROM channel_presence_events
    WHERE channel_type = 'web'
      AND channel_id = v_device_id;

    SELECT record_channel_presence(
        'web',
        v_device_id,
        p_presence_kind,
        'inbound',
        v_device_id,
        NULL,
        jsonb_strip_nulls(jsonb_build_object(
            'display_mode', NULLIF(btrim(COALESCE(p_display_mode, '')), ''),
            'visibility', NULLIF(btrim(COALESCE(p_visibility, '')), '')
        )),
        60
    )
    INTO v_event;

    RETURN v_event;
END;
$$;

COMMENT ON FUNCTION record_pwa_presence(TEXT, TEXT, TEXT, TEXT) IS
    'Coalesce one PWA device to its latest short-lived channel presence record.';
