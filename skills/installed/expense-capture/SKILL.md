---
name: expense-capture
description: Capture a receipt or purchase as a structured, source-linked local expense record without guessing fields
category: productivity
requires:
  tools: [remember]
contexts: [chat]
aliases: [expense, expenses, receipt, receipts, reimbursement, purchase]
bound_tools: [recall, remember, email_search, email_read, read_file, manage_backlog, todoist_create_task, asana_create_task]
---

# Expense Capture

Turn purchase details into a consistent local record. This skill records information in Hexis; it does not claim to submit an expense to an accounting or reimbursement system.

## Workflow

1. **Identify the source.** Use details pasted by the user, an exact file path they provide, or a targeted email search they request. Do not search broad mail history or filesystem locations to hunt for receipts.
2. **Extract without invention.** Capture merchant, transaction date, amount, currency, tax or tip when explicit, category, payment method suffix when present, business purpose, reimbursable status, project or client, and a source reference.
3. **Mark unknowns.** Never infer currency from locale, use the email date as the transaction date, expand a partial card number, or invent a tax split. Ask only for missing fields that materially affect the user's purpose.
4. **Check likely duplicates.** Use `recall` for records with the same merchant, date, and amount. If one is similar, show it and ask whether to reuse, correct, or create a separate record.
5. **Confirm the record.** Present the normalized fields. An explicit request such as "capture this receipt" authorizes saving that shown record; an exploratory request does not.
6. **Store with provenance.** Use `remember` for a concise episodic record, including the source kind and reference. Avoid raw receipt bodies and unnecessary personal or payment data.
7. **Offer the next action.** If reimbursement or follow-up is needed, propose a local, Todoist, or Asana task. Create it only after the user chooses the task system and exact content.

## Control and Failure Boundaries

- Never claim an expense was filed, reimbursed, categorized by accounting, or synced externally unless a real tool reports that result.
- Never store full card or bank numbers, authentication data, or irrelevant personal details.
- If a file format cannot be read, preserve the partial structured record and ask for pasted text or a supported export. Say what failed and what the user can do next.
