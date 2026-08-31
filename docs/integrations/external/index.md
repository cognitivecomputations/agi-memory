<!--
title: External Integrations
summary: Third-party service integrations for Hexis
read_when:
  - "You want to connect external services"
  - "You want to see what integrations are available"
section: integrations
-->

# External Integrations

Connect Hexis to external services for everyday life, calendar, email, CRM,
productivity, search, and media.

## Quick Links

| Page | Services |
|------|----------|
| [Everyday Life](everyday-life.md) | Notion, Spotify, Home Assistant, Open-Meteo weather, Trello |
| [Calendar and Email](calendar-and-email.md) | Google Calendar, SMTP email, SendGrid |
| [CRM](crm.md) | HubSpot, Contacts system |
| [Productivity](productivity.md) | Todoist, Asana |
| [Search and Web](search-and-web.md) | Brave Search, Firecrawl, web tools |
| [Media](media.md) | YouTube, Twitter/X, Fathom, image/video generation |

## Configuration Pattern

Most external integrations require an API key:

```bash
# Set the API key
hexis tools set-api-key <key_name> <value>

# Enable the tool
hexis tools enable <tool_name>

# Verify
hexis tools status
```

API keys are stored as environment variable names in the database config, not as raw values.

## Available Tools by Service

| Service | Tools | Energy Cost |
|---------|-------|-------------|
| Notion | `notion_search`, `notion_get_page`, `notion_query_data_source`, `notion_create_page` | 2-3 |
| Spotify | `spotify_search`, `spotify_playback_state`, `spotify_control_playback` | 1-3 |
| Home Assistant | `home_assistant_states`, `home_assistant_call_service` | 1-3 |
| Open-Meteo weather | `weather_forecast` | 1 |
| Trello | `trello_list_boards`, `trello_list_cards`, `trello_create_card`, `trello_update_card` | 1-3 |
| Google Calendar | `calendar_events`, `calendar_create`, `calendar_update`, `calendar_delete`, `meeting_prep` | 2-4 |
| Email (SMTP) | `email_send`, `email_list`, `email_read`, `email_search`, `email_forward` | 2-4 |
| Email (SendGrid) | `email_send_sendgrid` | 4 |
| HubSpot | `hubspot_*` | 1 |
| Todoist | `todoist_create`, `todoist_complete` | 2 |
| Asana | `asana_create` | 2 |
| Brave Search | `brave_search` | 2 |
| Firecrawl | `firecrawl_scrape` | 3 |
| YouTube | `youtube_*` | 1 |
| Twitter/X | `twitter_search` | 2 |
| Fathom | `fathom_transcripts`, `fathom_ingest` | 2-4 |
| DALL-E | `generate_image` | 3 |
| Stability AI | `generate_image` | 3 |
| Runway ML | `generate_video` | 8 |
