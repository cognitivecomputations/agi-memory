---
name: spotify
description: Search Spotify, inspect current playback and devices, and control playback with explicit approval
category: creative
requires:
  tools: [spotify_search, spotify_playback_state]
contexts: [chat, heartbeat]
bound_tools: [spotify_search, spotify_playback_state, spotify_control_playback]
---

# Spotify

Use this for catalog discovery and the user's connected Spotify player.

## Principles

- Search results and playback state are task-scoped provider data. Do not ingest Spotify content into model training or long-term datasets.
- Read playback state before choosing a device or assuming one is active.
- `spotify_control_playback` changes provider state and always goes through approval. Name the action, device when relevant, and track/context URI before calling it.
- Playback controls may require Spotify Premium. Surface the provider's reason and recovery step when an account or device cannot perform the action.
- Queue takes a Spotify URI; transfer takes a device ID; repeat accepts `off`, `context`, or `track`.
