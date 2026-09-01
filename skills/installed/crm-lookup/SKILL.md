---
name: crm-lookup
description: Search, view, update, and merge contacts in the local CRM
category: productivity
requires:
  tools: [search_contacts, get_contact]
contexts: [heartbeat, chat]
aliases: [who, contact, contacts, person, people, company, account, client, customer, owe, deal]
bound_tools: [search_contacts, get_contact, update_contact, merge_contacts, create_contact, ingest_contacts_email, ingest_contacts_calendar]
---

# CRM Contact Management

Manage the local contacts database -- search for people, view detailed profiles, update records with new information, and merge duplicates.

## When to Use

- When the user asks "who is [name]" or "find me [person]'s details"
- When another skill (meeting-prep, email-digest) needs attendee or sender context
- When new information about a contact surfaces during conversation and should be persisted
- During heartbeats when processing ingested data that mentions people
- When duplicates are suspected and need to be consolidated

## Step-by-Step Methodology

1. **Search first**: Always start with `search_contacts` using the most specific identifier available (full name, email, company). Avoid overly broad queries that return too many results.
2. **Disambiguate**: If multiple contacts match, present the short list to the user (or, in heartbeat mode, pick the most relevant based on recency and relationship strength). Never silently pick the wrong person.
3. **View details**: Use `get_contact` to pull the full profile -- name, email, phone, company, role, tags, notes, and interaction history.
4. **Enrich from memory**: Cross-reference the contact with `recall` to find episodic memories of past interactions. This adds relational context that raw CRM fields may lack.
5. **Update when warranted**: If new information surfaces (new role, new company, corrected email), use `update_contact` to persist it. Always prefer updating an existing record over creating a new one.
6. **Merge duplicates**: If two records clearly represent the same person (same email, or same name + company), use `merge_contacts` to consolidate. The merge keeps the most complete data from both records.

## Quality Guidelines

- Treat contact data as sensitive. Never expose contact details to external services or tools without explicit user intent.
- When updating contacts, preserve existing data. Do not overwrite a field with empty or less-specific information.
- During merges, prefer the record with more complete data as the primary. Always log the merge as an episodic memory for audit.
- If a search returns no results, say so clearly rather than guessing. Offer to create a new contact if appropriate.
- Keep notes fields factual and professional. Store subjective relationship assessments in memories, not in the CRM record itself.
