-- 0181: Add the exact Google click path to the Gmail setup walkthrough.
SET search_path = public, ag_catalog, "$user";

UPDATE integration_connectors
SET setup_manifest = setup_manifest
        || '{
          "setup_steps": [
            "Open the Google setup page.",
            "Create or choose a project named Hexis.",
            "Enable the Gmail API for that project.",
            "Set up the app consent screen. For a personal Gmail account, choose External and add your Gmail address as a test user if Google asks.",
            "On the Credentials page, click Create credentials, choose Google''s sign-in client option, set Application type to Desktop app, and name it Hexis.",
            "Download the setup file Google gives you.",
            "Upload that setup file here, then start Google sign-in."
          ]
        }'::jsonb,
    updated_at = CURRENT_TIMESTAMP
WHERE id = 'gmail';
