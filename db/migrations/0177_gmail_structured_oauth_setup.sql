-- 0177: Treat Gmail OAuth setup as a first-class UI flow with credential modes.
SET search_path = public, ag_catalog, "$user";

UPDATE integration_connectors
SET setup_manifest = setup_manifest
        || '{
	          "requires_user_client_secret": false,
	          "preferred_credential_mode": "hosted_oauth",
	          "credential_modes": ["hosted_oauth", "advanced_self_hosted", "upload_json", "path", "configured_env"],
	          "hosted_oauth_configured": false,
	          "setup_label": "Gmail sign-in",
	          "setup_steps": [
	            "Open the Google setup page.",
	            "Create or choose a project named Hexis.",
	            "Enable the Gmail API for that project.",
	            "Set up the app consent screen. For a personal Gmail account, choose External and add your Gmail address as a test user if Google asks.",
	            "On the Credentials page, click Create credentials, choose Google''s sign-in client option, set Application type to Desktop app, and name it Hexis.",
	            "Download the setup file Google gives you.",
	            "Upload that setup file here, then start Google sign-in."
	          ],
	          "technical_next_step": "For developers and hosted builds: configure HEXIS_GMAIL_OAUTH_CLIENT_ID and HEXIS_GMAIL_OAUTH_CLIENT_SECRET in the Hexis API environment to make built-in Google sign-in available."
	        }'::jsonb
	        || jsonb_build_object(
	          'notes',
	          '[
	            "Use the structured Gmail setup UI to choose provider powers and email memory policy before Google sign-in.",
	            "The default user path is built-in Google sign-in when configured for the build.",
	            "If this local build needs setup, the UI should walk the user through Google Cloud step by step and accept the downloaded setup file.",
	            "CLI surfaces can accept a local JSON path directly after the guide explains what file to download.",
	            "Paste the full localhost redirect URL back into the conversation.",
	            "Long-lived tokens are stored outside Postgres in the private auth store."
	          ]'::jsonb
        ),
    updated_at = CURRENT_TIMESTAMP
WHERE id = 'gmail';
