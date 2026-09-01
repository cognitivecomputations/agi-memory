---
name: notion
description: Search, read, query, and create content in the connected Notion workspace
category: productivity
requires:
  tools: [notion_search, notion_get_page, notion_query_data_source]
contexts: [chat, heartbeat]
bound_tools: [notion_search, notion_get_page, notion_query_data_source, notion_create_page]
---

# Notion

Use this for pages and data sources explicitly shared with the connected Notion
integration.

## Principles

- Search first when an exact page or data-source ID is unknown.
- A retrieved page contains properties; request its blocks when the task needs page content.
- Use current data-source IDs, not legacy database IDs, for queries and new database rows.
- Treat filters, sorts, properties, and children as native Notion JSON and keep them no larger than the requested operation needs.
- `notion_create_page` is an external write. Summarize the parent and content, then let the normal approval gate obtain the user's explicit choice.
- Never infer access to unshared workspace content and never ask for a Notion token in chat.
