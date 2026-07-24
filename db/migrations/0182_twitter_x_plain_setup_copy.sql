-- 0182: Keep Twitter/X visible setup copy out of protocol jargon.
SET search_path = public, ag_catalog, "$user";

UPDATE integration_connectors
SET setup_manifest = setup_manifest
        || jsonb_build_object(
          'user_next_step',
          'Create or choose an X Developer app, enable user sign-in, register http://localhost:1 as the callback URI, then start Twitter/X connection setup. Request only the capabilities you want; archive import is still available through a local export path.'
        ),
    updated_at = CURRENT_TIMESTAMP
WHERE id = 'twitter_x';
