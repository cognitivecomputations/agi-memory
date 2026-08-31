---
name: home-assistant
description: Read Home Assistant entity states and call explicit services on a connected home
category: productivity
requires:
  tools: [home_assistant_states]
contexts: [chat, heartbeat]
bound_tools: [home_assistant_states, home_assistant_call_service]
---

# Home Assistant

Use this for the connected Home Assistant instance.

## Principles

- Read the target entity state before acting when current state affects the decision.
- A service call must specify the exact domain and service. Prefer a concrete `entity_id` over a broad service target.
- `home_assistant_call_service` changes the physical or digital home and always goes through approval. Explain the intended real-world effect first.
- Never ask for a long-lived token in chat. Setup stores only the environment variable name the user selected.
- If Home Assistant returns an unexpected set of changed states, report them; do not silently retry a state-changing call.
