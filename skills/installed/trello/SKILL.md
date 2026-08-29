---
name: trello
description: List Trello boards, lists, and cards and create or update cards with explicit approval
category: productivity
requires:
  tools: [trello_list_boards, trello_list_cards]
contexts: [chat, heartbeat]
bound_tools: [trello_list_boards, trello_list_cards, trello_create_card, trello_update_card]
---

# Trello

Use this for the connected Trello member's boards and cards.

## Principles

- List boards and lists first when the target list ID is unknown.
- List cards from exactly one board or list per call.
- Creating or updating a card is an external write and always goes through approval. Show the target list/card and intended fields first.
- Updating supports only `name`, `desc`, `due`, `dueComplete`, `closed`, `idList`, and `pos`; do not invent other fields.
- Never ask for a Trello API key or token in chat. Setup stores only the two environment variable names the user selected.
