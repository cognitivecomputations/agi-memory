---
name: personal-notes
description: Search, read, create, and carefully edit Markdown notes in a user-mounted Obsidian vault or Bear export
category: knowledge
requires:
  tools: [read_file, glob]
contexts: [chat]
aliases: [notes, obsidian, bear, vault, markdown, notebook]
bound_tools: [glob, grep, list_directory, read_file, write_file, edit_file, fast_ingest, recall, remember]
---

# Personal Notes

Work with notes the user intentionally exposes inside the Hexis workspace. This supports Markdown-based Obsidian vaults and exported Bear notes; it does not reach into an unmounted home directory or Bear's private live database.

## Establish the Note Root

1. Ask for the exact mounted vault or export path when it is not already explicit in the conversation or workspace configuration.
2. Confirm that the path is inside the permitted workspace. Never guess a path, scan the user's home directory, or reuse an unrelated repository because it happens to contain Markdown.
3. Use a narrow `glob`, `list_directory`, or `grep` under that root. Expand the search only when the initial results justify it.

## Read and Search

- Search titles, tags, links, and content, then read only the most relevant notes.
- Cite note paths in the answer so the user can locate the source.
- Preserve the distinction between text in a note, a Hexis memory, and the assistant's inference.
- Do not ingest an entire vault into memory by default. Use `fast_ingest` only for files the user explicitly chooses and explain that it creates derived memories.

## Create and Edit

1. Resolve the exact target path and inspect any existing file before proposing a change.
2. Preserve YAML frontmatter, tags, wiki links, embeds, heading style, and surrounding formatting.
3. Prefer a targeted `edit_file` operation for an existing note. For a new note, show the proposed path and content before `write_file` unless the user's request already specified both.
4. Never overwrite a note to append a small section, never silently rename or move notes, and never resolve link conflicts by deleting content.
5. After a write, state the path and what changed. If the tool reports a failure, do not claim the note was saved.

## No Dead Ends

- If the vault is not mounted, tell the user to mount or copy it into the Hexis workspace and provide that path.
- For Bear, request a Markdown export or mounted export directory; do not advise modifying Bear's internal database.
- If writes are disallowed, provide the ready-to-paste Markdown and exact intended path so the work remains useful.
