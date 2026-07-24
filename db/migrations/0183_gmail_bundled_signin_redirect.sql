-- 0183: Align Gmail connector manifest with bundled Desktop sign-in.
SET search_path = public, ag_catalog, "$user";

UPDATE integration_connectors
SET setup_manifest = setup_manifest
        || '{
          "redirect_uri": "http://localhost",
          "preferred_credential_mode": "hosted_oauth",
          "setup_label": "Gmail sign-in"
        }'::jsonb,
    updated_at = CURRENT_TIMESTAMP
WHERE id = 'gmail';
