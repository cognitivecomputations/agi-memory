---
name: weather
description: Get current conditions and up to sixteen days of forecast for a place or the connected default location
category: research
requires:
  tools: [weather_forecast]
contexts: [chat, heartbeat]
bound_tools: [weather_forecast]
---

# Weather

Use this for current conditions and forecasts from Open-Meteo.

## Principles

- Use the named location from the request. With no location, use the verified connected default.
- If a place is ambiguous, state which geocoding match was used; setup records the match and coordinates.
- Forecasts are estimates. Preserve provider units and timestamps and distinguish current conditions from daily extrema.
- Request only the number of days the task needs, from one through sixteen.
