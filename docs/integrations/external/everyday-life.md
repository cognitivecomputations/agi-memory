<!--
title: Everyday-Life Integrations
summary: Connect cloud services and use private Mac apps through a paired node
read_when:
  - "You want to connect Notion, Spotify, Home Assistant, weather, Trello, or private Mac apps"
  - "You need to understand Wave B connector permissions or credential storage"
  - "You want to use Apple Reminders, Notes, Calendar, Shortcuts, or 1Password"
section: integrations
-->

# Everyday-Life Integrations

Hexis has first-class connectors and skills for Notion, Spotify, Home Assistant,
Open-Meteo weather, and Trello. Open **Connections** in the web dashboard, choose
the smallest capability set you need, and follow the provider-specific setup card.
The same guided flow is available in chat.

Except for Spotify's non-secret app client ID, never paste a provider credential
into Hexis chat or the dashboard. Put the credential in the environment of the
Hexis process that will use it, then enter only that environment variable's name.
Postgres stores the chosen name and connection metadata, not the secret value.
Spotify OAuth tokens live in Hexis's mode-0600 private auth store.

Apple Reminders, Notes, Calendar, Shortcuts, screenshots, and 1Password are
different: they stay on a signed companion Mac rather than becoming cloud
connectors. Pair and run that outward-only device with `hexis node`; see
[Companion Nodes](../../operations/companion-nodes.md).

## Providers

| Provider | Setup | Read capabilities | Approval-gated changes |
|----------|-------|-------------------|-------------------------|
| Notion | Internal integration token plus pages/data sources shared with it | Search, read pages/blocks, query data sources | Create a page |
| Spotify | Authorization Code with PKCE and a Spotify app client ID | Catalog search, playback state, devices | Play, pause, queue, seek, volume, transfer, shuffle, repeat |
| Home Assistant | Base URL plus long-lived access token | Entity states | Call a specific service |
| Weather | Verified default place; no credential | Current conditions and 1–16 day forecasts | None |
| Trello | Power-Up API key and authorized token | Boards, lists, cards | Create or update a card |

Provider access and permission to change provider state are separate. Selecting a
write capability during connection makes that operation available, but each actual
write still goes through Hexis's normal approval gate.

## Notion

1. Create an internal integration in Notion.
2. Share only the pages or data sources Hexis may use with that integration.
3. Put the token in an environment variable of your choosing.
4. In **Connections → Notion**, enter that variable's name, choose capabilities,
   and verify.

Hexis uses the API version declared by the live connector manifest. Queries use
current Notion data-source IDs. A successful token does not grant workspace-wide
access; content must also be shared with the integration.

## Spotify

1. Create a Spotify app.
2. Start Spotify setup in Hexis and copy the exact redirect URI it displays into
   the app's redirect-URI allowlist. The local default is
   `http://127.0.0.1:43817/api/integrations/spotify/callback`; do not substitute
   `localhost`.
3. Enter the app client ID directly (it is not secret), or explicitly choose an
   environment variable containing it.
4. Open the returned sign-in URL. Spotify normally completes the local callback
   automatically; the setup card also accepts the full callback URL as recovery.

Playback controls may require Spotify Premium. Disconnecting Spotify removes the
saved access and refresh tokens.

## Home Assistant

Create a long-lived access token in the Home Assistant profile, store it in an
environment variable, then provide its variable name and the complete Home
Assistant URL. Hexis verifies the REST API before saving the connection. Service
calls name the exact domain, service, and preferably the target entity; Hexis does
not silently retry state-changing calls.

## Weather

Weather uses Open-Meteo and needs no API key. Enter a default city or place. Hexis
geocodes it, verifies a forecast for the best match, saves the matched coordinates,
and shows the place it selected. A forecast request can always name another place
without changing the saved default.

## Trello

Create or select a Trello Power-Up, obtain its API key, and authorize a token with
only the read/write powers you chose. Store the key and token in two environment
variables and enter only those variable names in Hexis. List boards and lists before
creating a card when the target list ID is unknown.

## Private Mac apps and 1Password

After a Mac is paired, the `host-node` skill provides structured tools for
Reminders, Notes, Calendar, and Shortcuts. Every invocation is approval-gated,
including reads. Hexis uses fixed local automation programs; chat text is never
evaluated as AppleScript or shell input. The node advertises only the capabilities
supported by executables present on that Mac, and adding one later requires a new
capability approval.

The 1Password tool lists only redacted item metadata. Secret fields are addressed
with an exact `op://vault/item/field` reference and copied directly to that Mac's
clipboard. The value does not cross the node gateway or enter model context, and
the node clears it after the approved interval only when the clipboard has not
changed in the meantime.

## Troubleshooting and disconnection

- If setup says a selected environment variable is not set, add it to the runtime
  environment, restart the relevant Hexis process, and press **Verify** again.
- If Notion returns no content, share the target page or data source with the
  integration.
- If Spotify rejects the redirect, copy the exact URI from the current setup card
  and use a loopback IP for HTTP callbacks.
- If Home Assistant is unreachable, verify that its URL is reachable from the Hexis
  process, not only from your browser.
- If weather matched the wrong place, disconnect it and reconnect with a more
  specific city, region, or country.
- If Trello denies a write, reauthorize a token with the selected write power.

Use **Connections → Disconnect** to revoke a saved connection. This is an explicit,
confirmed action; Hexis never disconnects a provider on a timer.

## Related

- [Skills](../../guides/skills.md)
- [Tools Reference](../../reference/tools.md)
- [Environment Variables](../../operations/environment-variables.md)
- [Companion Nodes](../../operations/companion-nodes.md)
