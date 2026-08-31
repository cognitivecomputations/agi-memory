---
name: github-issues
description: Search, read, create, and comment on GitHub issues through the GitHub MCP server.
category: productivity
contexts: [heartbeat, chat]
mcp:
  server: github
  command: npx
  args: ["-y", "@modelcontextprotocol/server-github"]
  env_requires: [GITHUB_PERSONAL_ACCESS_TOKEN]
bound_tools:
  - mcp_github_search_issues
  - mcp_github_list_issues
  - mcp_github_get_issue
  - mcp_github_create_issue
  - mcp_github_add_issue_comment
  - mcp_github_update_issue
---

# GitHub Issues

Work with GitHub issues on repositories the configured token can reach. This skill
binds the GitHub MCP server: activating it starts the server (first use only) and
unlocks the bound tools for this turn.

## Workflow

1. **Search before creating.** Use `mcp_github_search_issues` to check whether the
   issue already exists; duplicate issues waste maintainer attention. Comment on the
   existing issue instead when appropriate.
2. **Creating an issue is an outward-facing action.** Confirm with the user before
   filing unless they explicitly asked you to file it. Show the title and body you
   intend to submit.
3. Reference concrete evidence (file paths, line numbers, reproduction steps) that
   you actually verified this turn.
4. After creating, report the issue URL from the tool result — never invent one.

## If activation reports needs_setup

Relay the exact `next_step` to the user (#102). It will point them to run
`hexis auth github setup-token` in a terminal — never ask for or accept a
token pasted into chat; the tool activation path already checks Hexis's own
stored credential automatically on retry. Explain, if useful, why a public
repo still needs this: reading it is public, but creating or commenting on
an issue as a specific account is not. Do not claim GitHub capability is
missing — it is installed and one step away.
