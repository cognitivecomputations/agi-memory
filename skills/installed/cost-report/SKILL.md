---
name: cost-report
description: Query and report on API usage costs across LLM, embedding, and tool providers
category: analytics
requires:
  tools: [query_usage]
contexts: [heartbeat, chat]
aliases: [cost, costs, spend, spent, usage, tokens, budget, bill, billing]
bound_tools: [query_usage]
---

# Usage Cost Reporting

Query the API usage ledger and produce clear, actionable cost reports broken down by provider, model, and time period.

## When to Use

- When the user asks "how much have I spent" or "show me my usage"
- During heartbeats to check if spending is approaching a configured budget threshold
- When evaluating whether to switch models or providers based on cost efficiency
- As part of a daily or weekly briefing that includes operational metrics
- After a heavy ingestion or research session to assess the cost impact

## Step-by-Step Methodology

1. **Determine the time window**: Default to the current billing period (calendar month) unless the user specifies a range. Common windows: today, this week, this month, last 30 days, custom range.
2. **Query the ledger**: Use `query_usage` with the appropriate date range. The tool returns rows from the `api_usage` table with provider, model, token counts (input/output), and computed cost.
3. **Aggregate by dimension**: Break down the raw data into useful summaries:
   - **By provider**: Total cost per provider (OpenAI, Anthropic, local inference, etc.)
   - **By model**: Cost per model to identify which models consume the most budget
   - **By category**: Separate LLM inference costs from embedding costs from tool API costs
   - **By day**: Daily trend to spot spikes or patterns
4. **Compute key metrics**: Calculate:
   - Total spend for the period
   - Average daily cost
   - Cost per conversation (if trackable)
   - Projected monthly cost at current burn rate
5. **Compare to budget**: If a budget threshold is configured, compare current spend and projection against it. Flag if on track to exceed.
6. **Present the report**: Format as a clean table or summary. Lead with the headline number (total spend), then break down by the most useful dimension.

## Quality Guidelines

- Always show actual numbers, not vague descriptions. "You have spent $4.32 this month" is useful; "you have moderate usage" is not.
- Round costs to two decimal places for readability.
- When projecting monthly cost, note that it is an extrapolation and actual spend may vary.
- If the usage ledger is empty or the query returns no data, say so plainly rather than reporting zeros that could be misleading.
- In heartbeat context, only generate a cost report if spend is approaching a threshold or if a goal explicitly asks for cost monitoring. Do not waste energy on routine cost checks every heartbeat.
- Protect cost data as operational metadata. Do not expose it to external services.
