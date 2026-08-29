<!--
title: Outbound Safety
summary: Audit and control purpose-bound contact, cadence, disclosure, and opt-outs
read_when:
  - "You want to understand or pause autonomous messages"
  - "A recipient sent STOP"
  - "You want to audit why Hexis contacted someone"
section: guides
-->

# Outbound Safety

Hexis applies one database-owned safety contract to every person-facing delivery:
provider tools, the formal outbox, and replies sent directly through a channel
adapter. A transport cannot opt out of the contract.

## Inspect and Pause

Open **Outbound** in the dashboard. The page shows:

- every attempted communication, its backed purpose, attention cost, disclosure
  form, and final delivery state;
- each person's per-channel points, regeneration, observed cadence, reciprocity,
  unanswered count, and overdraft strain;
- global and per-person pause controls.

**Pause all outbound** stops every delivery, including replies. **Pause this contact**
stops one person. Both are reversible and take effect without a worker restart.
Neither control can erase a recipient's opt-out.

## Purpose and Attention

Third-party contact must refer to an existing goal, active responsibility, inbound
reply thread, or explicit user request. The primary user may also be contacted for
connection. Missing or unbacked purposes fail before a provider is called.

Only unsolicited third-party contact spends attention points. Replies are free.
Inbound engagement restores points, while repeated silence slows or stops further
non-urgent outreach. A genuinely urgent, backed message may overdraft; the resulting
strain remains visible and suppresses ordinary contact until it recovers. Work on a
user-assigned goal is cheaper, but never free to the recipient's inbox.

## Disclosure and STOP

The first third-party message, a new thread, or contact after a long gap receives the
full agent/principal identity plus literal STOP instructions. Later messages retain a
short AI marker. The primary user never receives this disclosure.

`STOP`, `UNSUBSCRIBE`, `OPT OUT`, `EXCOMMUNICATE`, and punctuation variants block the
person immediately across all known channels. Hexis acknowledges the first STOP once,
notifies the primary user through their inbox, records the evidence, and then remains
silent. Repeated STOP messages receive no further reply. Only that recipient sending
`START` or `UNSTOP` reverses the block.

## Tool Contract

Custom person-facing tools must declare `ToolSpec.outbound`. Registration fails for a
messaging send/reply/post without the descriptor. The tool schema then requires:

- `purpose_kind`: `goal`, `responsibility`, `reply`, `user_request`, or `connection`;
- `purpose_reference`: the durable record or trusted turn backing that purpose;
- optional `urgency`: `low`, `normal`, `high`, or `urgent`.

The dispatcher checks STOP and pause controls before asking for approval, then repeats
the full purpose/budget/control authorization immediately before the provider call.
This preserves both recipient silence and race safety.

## Recovery

If a delivery is denied, open **Outbound** and read the recorded reason. Budget denials
show the current points or strain; pause denials identify the control to resume. A
STOP denial cannot be overridden by a goal, urgency, approval, or operator pause
control. Ask the recipient to contact you outside Hexis if they want to opt back in;
Hexis must not send that request for you.

## Related

- [Channels Setup](channels-setup.md)
- [Config Keys](../reference/config-keys.md)
- [Heartbeat](heartbeat.md)
