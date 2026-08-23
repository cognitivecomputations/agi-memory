---
name: research
description: How to research topics using web tools
requires:
  tools: [web_search, web_fetch]
  config: [tavily]
contexts: [heartbeat, chat]
aliases: [research, look, find, search, investigate, learn, read, web, online, news]
bound_tools: [web_search, web_fetch, recall, remember, web_summarize, brave_search, firecrawl_scrape]
---

# Research Methodology

When investigating a topic that requires current or external information:

1. Use `web_search` for broad queries to find relevant sources
2. Use `web_fetch` to read the most promising results in full
3. Cross-reference findings with existing memories via `recall`
4. Store important findings with `remember` for future reference

## When to Research

- When a goal requires information you don't have in memory
- When existing memories have low confidence or are outdated
- When the user asks about recent events or facts
- When you need to verify something before acting on it

## Research Quality

- Use specific, targeted search queries rather than broad ones
- Cross-reference multiple sources before accepting a claim
- Note the source URL when storing findings as memories
- Prefer authoritative sources (official docs, academic, established media)
