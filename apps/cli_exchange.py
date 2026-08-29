"""Command-line I/O and reporting for Hexis Memory Exchange (HMX)."""

from __future__ import annotations

import asyncio
import copy
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import asyncpg

from core.hmx_files import (
    load_hmx_file,
    parse_hmx_text,
    serialize_hmx_document,
    write_private_hmx_file,
)

from core.memory_exchange import (
    HmxAnalysisResult,
    HmxAuthoritativeResult,
    HmxDryRunResult,
    HmxImportResult,
    HmxPolicyError,
    HmxStagingResult,
    accept_staged_import,
    default_import_strategy,
    demote_staged_to_analysis,
    dry_run_hmx,
    export_hmx,
    import_hmx,
    normalize_replace_sections,
    pending_hmx_reviews,
    promote_analysis_to_staged,
    quote_staged_import,
    reject_staged_import,
    modify_staged_import,
    verify_mind_continuity,
)
from core.protected_replacement import (
    operator_override_signing_material,
    operator_override_signing_payload,
    validate_operator_override_fields,
    validate_operator_override_reason_state,
)
from core.trust_anchors import (
    HMX_OPERATOR_PUBLIC_KEY_ENV,
    load_trust_anchor_verifier_from_env,
)


def _csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    values = [part.strip() for part in value.split(",") if part.strip()]
    return values or []


def _section_csv(value: str | None) -> list[str] | None:
    values = _csv(value)
    return [item.replace("-", "_") for item in values] if values is not None else None


async def _connect(dsn: str, wait_seconds: int) -> asyncpg.Connection:
    deadline = time.monotonic() + max(wait_seconds, 1)
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        conn: asyncpg.Connection | None = None
        try:
            conn = await asyncpg.connect(dsn, ssl=False, command_timeout=60.0)
            await conn.execute("LOAD 'age'")
            await conn.execute('SET search_path = public, ag_catalog, "$user"')
            return conn
        except Exception as exc:  # pragma: no cover - exact driver errors vary
            last_error = exc
            if conn is not None:
                await conn.close()
            await asyncio.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    raise TimeoutError(
        f"failed to connect to Postgres after {wait_seconds}s: {last_error}"
    )


def _timestamp(value: str | None, flag: str) -> datetime | None:
    if value is None:
        return None
    candidate = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise HmxPolicyError(
            f"{flag} must be an ISO 8601 date or timestamp; got {value!r}"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _load_document(path_value: str) -> dict[str, Any]:
    if path_value == "-":
        return parse_hmx_text(sys.stdin.read(), label="stdin")
    return load_hmx_file(path_value)


def _serialized_export(document: dict[str, Any], output_format: str) -> str:
    return serialize_hmx_document(document, output_format)


def _write_private_file(path_value: str, content: str, *, overwrite: bool) -> Path:
    return write_private_hmx_file(path_value, content, overwrite=overwrite)


def _mind_default_output(document: dict[str, Any], output_format: str) -> Path:
    """Choose a private, collision-resistant path from the exported truth."""

    from core.config import hexis_home

    output_dir = hexis_home() / "exports"
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    source = document.get("source") or {}
    instance = "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in str(source.get("instance_id") or "hexis")
    ).strip("_")[:40] or "hexis"
    exported_at = str(document.get("exported_at") or "")
    try:
        stamp = datetime.fromisoformat(exported_at.replace("Z", "+00:00")).strftime(
            "%Y%m%d_%H%M%S"
        )
    except ValueError:
        stamp = "export"
    export_suffix = str(document.get("export_id") or "mind").removeprefix("exp_")[-8:]
    extension = "jsonl" if output_format == "jsonl" else "json"
    return output_dir / f"{instance}-mind-{stamp}-{export_suffix}.hmx.{extension}"


def _apply_skips(document: dict[str, Any], skipped: list[str]) -> dict[str, Any]:
    prepared = copy.deepcopy(document)
    sections = prepared.get("sections")
    if not isinstance(sections, dict):
        return prepared
    for section in skipped:
        sections.pop(section, None)
    return prepared


def _print_dry_run(
    result: HmxDryRunResult,
    *,
    as_json: bool,
    skipped: list[str],
    operator_override: dict[str, Any] | None = None,
) -> None:
    payload = asdict(result)
    payload["skipped_sections"] = skipped
    if operator_override is not None:
        payload["operator_override"] = operator_override
    if as_json:
        sys.stdout.write(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
        )
        return

    from apps.cli_theme import console, make_table

    console.print(
        f"[heading]HMX import dry run[/heading]  [muted]{result.export_id}[/muted]"
    )
    console.print(
        f"Intent [accent]{result.intent}[/accent]  Strategy [accent]{result.strategy}[/accent]  "
        f"Target [teal]{result.target_state.get('state', 'unknown')}[/teal]"
    )
    table = make_table("Section", ("Would import", {"justify": "right"}))
    hidden = {"total_records", "invalid_records", "duplicate_memories"}
    for section, count in result.counts.items():
        if section not in hidden and count:
            table.add_row(section, str(count))
    console.print(table)
    console.print(
        f"New embedding work: [accent]{result.estimated_embedding_items}[/accent]  "
        f"Duplicates: [warn]{len(result.duplicate_refs)}[/warn]  "
        f"Invalid records: [warn]{result.counts.get('invalid_records', 0)}[/warn]"
    )
    if skipped:
        console.print(f"Skipped by operator: [muted]{', '.join(skipped)}[/muted]")
    for conflict in result.conflicts:
        label = "conflict" if conflict.get("code") == "duplicate_content" else "blocked"
        style = "warn" if label == "conflict" else "fail"
        console.print(
            f"[{style}]{label}:[/{style}] {conflict.get('code')} "
            f"{json.dumps(conflict, sort_keys=True)}"
        )
    for warning in result.warnings:
        console.print(
            f"[warn]warning:[/warn] {warning.get('code')} {warning.get('error', '')}"
        )
    if result.can_import:
        console.print("[ok]This import is permitted by the current HMX policy.[/ok]")
    else:
        console.print(
            "[fail]This import cannot run with the selected intent and strategy.[/fail]"
        )
    if operator_override is not None:
        console.print("[fail]OPERATOR OVERRIDE SIGNING REQUEST[/fail]")
        anchor_id = operator_override.get("trust_anchor_id")
        if anchor_id:
            console.print(f"Trust anchor: [accent]{anchor_id}[/accent]")
        else:
            console.print(
                f"[fail]No trust anchor configured.[/fail] Set "
                f"[accent]{HMX_OPERATOR_PUBLIC_KEY_ENV}[/accent] to the base64 raw "
                "Ed25519 public key or PEM before execution."
            )
        console.print(
            "Payload SHA-256: "
            f"[accent]{operator_override['payload_sha256']}[/accent]"
        )
        console.print("Payload base64:")
        console.print(operator_override["payload_base64"])
        verification = operator_override["signature_verification"]
        console.print(
            "Signature: "
            f"[{('ok' if verification['status'] == 'verified' else 'warn')}]"
            f"{verification['status']}[/] - {verification['reason']}"
        )


def _print_import_result(
    result: (
        HmxImportResult | HmxAuthoritativeResult | HmxStagingResult | HmxAnalysisResult
    ),
    *,
    as_json: bool,
    skipped: list[str],
    continuity: dict[str, Any] | None = None,
) -> None:
    payload = asdict(result)
    payload["skipped_sections"] = skipped
    if continuity is not None:
        payload["continuity"] = continuity
    if as_json:
        sys.stdout.write(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
        )
        return

    from apps.cli_theme import console, make_table

    if isinstance(result, HmxAuthoritativeResult):
        overridden = [
            operation
            for operation in result.protected_operations
            if operation.get("replacement_executor") == "operator_override"
        ]
        pending = [
            operation
            for operation in result.protected_operations
            if operation.get("agent_acknowledgement_required")
        ]
        verified = [
            operation
            for operation in result.protected_operations
            if operation.get("disposition") == "verified_noop"
        ]
        resolved = [
            operation
            for operation in result.protected_operations
            if operation not in pending and operation not in verified
        ]
        if overridden:
            console.print(
                f"[fail]HMX OPERATOR OVERRIDE EXECUTED.[/fail] "
                f"[muted]{result.export_id}[/muted]"
            )
        else:
            console.print(
                f"[ok]HMX authoritative import processed.[/ok] "
                f"[muted]{result.export_id}[/muted]"
            )
        counts = result.inserted
        footer = (
            f"Protected requests awaiting agent decision: "
            f"[accent]{len(pending)}[/accent]  "
            f"verified no-ops: [accent]{len(verified)}[/accent]  "
            f"resolved: [accent]{len(resolved)}[/accent]"
        )
        for operation in result.protected_operations:
            disposition = str(operation.get("disposition") or "unknown")
            replacement_id = operation.get("replacement_id") or "no request id"
            section = operation.get("section") or "unknown section"
            if operation in pending:
                detail = (
                    "the next heartbeat will present accept, refuse, modify, "
                    "and defer choices"
                )
                style = "warn"
            elif operation in verified:
                detail = "content and trusted lineage already match; no write occurred"
                style = "ok"
            elif disposition == "refused":
                detail = (
                    "the prior agent refusal remains in force; revise the proposed "
                    "content or rationale before submitting a distinct request"
                )
                style = "fail"
            elif disposition == "modification_requested":
                detail = "the agent requested changes; submit a revised exchange and rationale"
                style = "warn"
            elif disposition == "executed":
                if operation.get("replacement_executor") == "operator_override":
                    detail = (
                        "OPERATOR OVERRIDE executed; agent acknowledgement was bypassed "
                        "and the normal reversion window remains open"
                    )
                    style = "fail"
                else:
                    detail = "this replacement was already executed; no duplicate write occurred"
                    style = "ok"
            elif disposition == "reverted":
                detail = (
                    "this replacement was executed and later reverted; revise the "
                    "proposal or rationale before submitting a distinct request"
                )
                style = "warn"
            else:
                detail = (
                    "no protected write occurred; inspect the JSON result for details"
                )
                style = "warn"
            console.print(
                f"[{style}]{disposition}:[/{style}] "
                f"{replacement_id} {section} - {detail}"
            )
    elif isinstance(result, HmxStagingResult):
        console.print(
            f"[ok]HMX import staged for review.[/ok] [muted]{result.export_id}[/muted]"
        )
        counts = result.staged
        footer = f"Pending review records: [accent]{len(result.staging_ids)}[/accent]"
    elif isinstance(result, HmxAnalysisResult):
        console.print(
            f"[ok]HMX import loaded into isolated analysis storage.[/ok] "
            f"[muted]{result.export_id}[/muted]"
        )
        counts = result.loaded
        footer = f"Analysis records: [accent]{len(result.analysis_ids)}[/accent]"
    else:
        console.print(
            f"[ok]HMX import complete.[/ok] [muted]{result.export_id}[/muted]"
        )
        counts = result.inserted
        footer = (
            f"Duplicates reused: [warn]{len(result.duplicate_refs)}[/warn]  "
            f"Warnings: [warn]{len(result.warnings)}[/warn]"
        )
        if result.work_summary:
            footer += (
                "  Work requeued: "
                f"[accent]{result.work_summary.get('requeued', 0)}[/accent]  "
                "failed preserved: "
                f"[warn]{result.work_summary.get('failed_preserved', 0)}[/warn]  "
                "retried: "
                f"[accent]{result.work_summary.get('retried', 0)}[/accent]  "
                "dropped: "
                f"[warn]{result.work_summary.get('dropped', 0)}[/warn]"
            )
    table = make_table("Section", ("Imported", {"justify": "right"}))
    for section, count in counts.items():
        table.add_row(section, str(count))
    console.print(table)
    console.print(footer)
    if skipped:
        console.print(f"Skipped by operator: [muted]{', '.join(skipped)}[/muted]")
    for warning in result.warnings:
        console.print(
            f"[warn]warning:[/warn] {warning.get('code')} {warning.get('error', '')}"
        )
    if continuity is not None:
        if continuity.get("verified"):
            console.print(
                "[ok]Mind continuity verified:[/ok] lineage and all constitutional "
                "section projections match the source file."
            )
        else:
            console.print(
                "[fail]Mind import completed, but continuity verification failed.[/fail] "
                + str(continuity.get("next_step") or "Inspect the JSON report.")
            )


async def run_export(dsn: str, args: Any) -> int:
    mind_export = bool(getattr(args, "mind", False))
    intent = args.intent
    if mind_export:
        if intent not in (None, "port"):
            raise HmxPolicyError("--mind uses intent=port and cannot be combined with another intent")
        if args.types or args.since or args.until:
            raise HmxPolicyError(
                "a mind export must be complete; use --intent port for a filtered exchange"
            )
        if args.redaction != "none":
            raise HmxPolicyError(
                "a mind export cannot redact constitutional state; use --intent port for a redacted exchange"
            )
        intent = "port"
    elif intent is None:
        raise HmxPolicyError(
            "export requires --intent port|duplicate|telepathy|analysis, or use --mind"
        )
    if args.include_raw:
        from apps.cli_theme import err_console

        err_console.print(
            "[warn]Raw units are sensitive source material; protect the exported file.[/warn]"
        )
    if args.redaction == "strict" and args.include_raw:
        raise HmxPolicyError("--redaction strict cannot be combined with --include-raw")
    since = _timestamp(args.since, "--since")
    until = _timestamp(args.until, "--until")
    if since and until and since > until:
        raise HmxPolicyError("--since must be earlier than or equal to --until")
    conn = await _connect(dsn, args.wait_seconds)
    try:
        document = await export_hmx(
            conn,
            intent=intent,
            include_protected=_section_csv(args.include_protected),
            include_raw_units=args.include_raw,
            include_config=args.include_config,
            include_in_flight_work=args.include_in_flight_work,
            include_audit_records=args.include_audit_records,
            types=_csv(args.types),
            since=since,
            until=until,
            redaction_policy=args.redaction,
            include_sensitive=getattr(args, "include_sensitive", False),
        )
    finally:
        await conn.close()

    if document.get("export_warnings"):
        from apps.cli_theme import err_console

        for warning in document["export_warnings"]:
            err_console.print(f"[warn]warning:[/warn] {warning}")

    content = _serialized_export(document, args.format)
    output = args.output
    if mind_export and output is None:
        output = str(_mind_default_output(document, args.format))
    if output in (None, "-"):
        sys.stdout.write(content)
    else:
        written = _write_private_file(output, content, overwrite=args.overwrite)
        from apps.cli_theme import console

        console.print(
            f"[ok]Exported HMX {document['hmx_version']} to {written}[/ok] "
            f"[muted]({document['statistics']['estimated_uncompressed_bytes']} bytes estimated)[/muted]"
        )
        if mind_export:
            console.print(
                "[muted]Next:[/muted] securely copy this private file to the empty "
                "destination, then run:\n"
                f"  hexis import {written.name} --mind --dry-run --json\n"
                f"  hexis import {written.name} --mind --confirm-intent port"
            )
    return 0


async def _prepare_operator_override_report(
    conn,
    *,
    document: dict[str, Any],
    forecast: HmxDryRunResult,
    args: Any,
) -> tuple[dict[str, Any], Any]:
    reason_code, evidence_ref = validate_operator_override_fields(
        acknowledgement=args.override_acknowledgement,
        reason_code=args.override_reason_code,
        evidence_ref=args.override_evidence_ref,
        rationale=str(args.replacement_rationale or ""),
    )
    await validate_operator_override_reason_state(conn, reason_code)
    operations = list(forecast.protected_policy.get("operations") or [])
    material = operator_override_signing_material(
        envelope=document,
        operations=operations,
        acknowledgement=args.override_acknowledgement,
        reason_code=reason_code,
        evidence_ref=evidence_ref,
        rationale=args.replacement_rationale,
        operator_identity=args.operator_identity,
    )
    try:
        verifier = load_trust_anchor_verifier_from_env()
    except ValueError as exc:
        raise HmxPolicyError(str(exc)) from exc
    anchor_id = getattr(verifier, "anchor_id", None)
    if args.operator_signature:
        payload = operator_override_signing_payload(
            envelope=document,
            operations=operations,
            acknowledgement=args.override_acknowledgement,
            reason_code=reason_code,
            evidence_ref=evidence_ref,
            rationale=args.replacement_rationale,
            operator_identity=args.operator_identity,
        )
        verification = verifier.verify_operator_signature(
            signature=args.operator_signature,
            payload=payload,
            operator_identity=args.operator_identity,
        )
        verification_payload = {
            "status": verification.status.value,
            "reason": verification.reason,
            "anchor_id": verification.anchor_id,
            "metadata": dict(verification.metadata),
        }
    else:
        verification_payload = {
            "status": "unverified",
            "reason": "no signature supplied; sign the exact payload before execution",
            "anchor_id": anchor_id,
            "metadata": {},
        }
    material.update(
        {
            "reason_code": reason_code,
            "evidence_ref": evidence_ref,
            "trust_anchor_environment": HMX_OPERATOR_PUBLIC_KEY_ENV,
            "trust_anchor_id": anchor_id,
            "signature_supplied": bool(args.operator_signature),
            "signature_verification": verification_payload,
            "ready_for_execution": verification_payload["status"] == "verified",
        }
    )
    return material, verifier


async def run_import(dsn: str, args: Any) -> int:
    document = _load_document(args.path)
    intent = str(document.get("export_intent") or "")
    mind_import = bool(getattr(args, "mind", False))
    if mind_import:
        if intent != "port":
            raise HmxPolicyError("--mind accepts only a port-intent HMX file")
        if args.strategy and args.strategy.replace("-", "_") != "additive":
            raise HmxPolicyError(
                "--mind uses the empty-target additive path; omit --strategy or choose additive"
            )
        if args.replace or args.force_replace:
            raise HmxPolicyError(
                "--mind never replaces an active target; use an ordinary authoritative import for explicit replacement review"
            )
        if args.skip_identity or args.skip_worldview or args.skip_narrative:
            raise HmxPolicyError(
                "--mind cannot skip constitutional sections; use an ordinary import for a partial transfer"
            )
    if args.confirm_intent and args.confirm_intent != intent:
        raise HmxPolicyError(
            f"intent confirmation mismatch: file declares {intent!r}, "
            f"but --confirm-intent was {args.confirm_intent!r}"
        )
    if not args.dry_run and not args.confirm_intent:
        raise HmxPolicyError(
            f"import requires --confirm-intent {intent}; run with --dry-run first to inspect it"
        )

    skipped = [
        section
        for section, enabled in (
            ("identity", args.skip_identity),
            ("worldview", args.skip_worldview),
            ("narrative", args.skip_narrative),
        )
        if enabled
    ]
    document = _apply_skips(document, skipped)
    strategy = (
        "additive" if mind_import else (args.strategy or default_import_strategy(intent))
    ).replace("-", "_")
    replace_sections = normalize_replace_sections(args.replace)
    skipped_replacements = sorted(set(skipped) & set(replace_sections))
    if skipped_replacements:
        raise HmxPolicyError(
            "a protected section cannot be both skipped and replaced: "
            + ", ".join(skipped_replacements)
        )
    if replace_sections and strategy != "authoritative":
        raise HmxPolicyError("--replace requires --strategy authoritative")
    override_arguments_present = any(
        (
            args.operator_signature,
            args.operator_identity,
            args.override_acknowledgement,
            args.override_reason_code,
            args.override_evidence_ref,
        )
    )
    if override_arguments_present and not args.force_replace:
        raise HmxPolicyError("operator override arguments require --force-replace")
    if args.force_replace and (strategy != "authoritative" or not replace_sections):
        raise HmxPolicyError(
            "--force-replace requires --strategy authoritative and at least one --replace section"
        )

    continuity: dict[str, Any] | None = None
    conn = await _connect(dsn, args.wait_seconds)
    try:
        forecast = await dry_run_hmx(
            conn,
            document,
            strategy=strategy,
            retry_failed_work=args.retry_failed_work,
            replace_sections=replace_sections,
            allow_locally_trusted_lineage=args.trust_matching_lineage_label,
        )
        operator_override = None
        verifier = None
        if args.force_replace and forecast.can_import:
            operator_override, verifier = await _prepare_operator_override_report(
                conn,
                document=document,
                forecast=forecast,
                args=args,
            )
        if args.dry_run:
            _print_dry_run(
                forecast,
                as_json=args.json,
                skipped=skipped,
                operator_override=operator_override,
            )
            invalid_supplied_signature = bool(
                operator_override
                and operator_override["signature_supplied"]
                and not operator_override["ready_for_execution"]
            )
            return 0 if forecast.can_import and not invalid_supplied_signature else 2
        if not forecast.can_import:
            _print_dry_run(
                forecast,
                as_json=args.json,
                skipped=skipped,
                operator_override=operator_override,
            )
            raise HmxPolicyError(
                "import blocked by preflight; resolve the reported policy conflict or strategy before retrying"
            )
        result = await import_hmx(
            conn,
            document,
            strategy=strategy,
            retry_failed_work=args.retry_failed_work,
            replace_sections=replace_sections,
            replacement_rationale=args.replacement_rationale,
            allow_locally_trusted_lineage=args.trust_matching_lineage_label,
            verifier=verifier,
            operator_signature=args.operator_signature,
            operator_identity=args.operator_identity,
            force_replace=args.force_replace,
            override_acknowledgement=args.override_acknowledgement,
            override_reason_code=args.override_reason_code,
            override_evidence_ref=args.override_evidence_ref,
        )
        if mind_import:
            continuity = await verify_mind_continuity(conn, document)
    finally:
        await conn.close()

    _print_import_result(
        result,
        as_json=args.json,
        skipped=skipped,
        continuity=continuity,
    )
    return 0 if continuity is None or continuity.get("verified") else 2


async def run_review(dsn: str, args: Any) -> int:
    conn = await _connect(dsn, args.wait_seconds)
    try:
        if args.review_command in (None, "list"):
            result: Any = await pending_hmx_reviews(conn)
        elif args.review_command == "accept":
            result = asdict(
                await accept_staged_import(
                    conn, args.staging_id, rationale=args.rationale
                )
            )
        elif args.review_command == "reject":
            result = asdict(
                await reject_staged_import(
                    conn, args.staging_id, rationale=args.rationale
                )
            )
        elif args.review_command == "modify":
            try:
                changes = json.loads(args.changes)
            except json.JSONDecodeError as exc:
                raise HmxPolicyError(
                    f"--changes must be a JSON object: {exc.msg}"
                ) from exc
            if not isinstance(changes, dict):
                raise HmxPolicyError("--changes must be a JSON object")
            result = asdict(
                await modify_staged_import(
                    conn,
                    args.staging_id,
                    changes,
                    modification_kind=args.modification_kind,
                    rationale=args.rationale,
                )
            )
        elif args.review_command == "quote":
            result = asdict(
                await quote_staged_import(
                    conn, args.staging_id, rationale=args.rationale
                )
            )
        elif args.review_command == "promote":
            staging_id = await promote_analysis_to_staged(
                conn, args.analysis_id, rationale=args.rationale
            )
            result = {"decision": "promoted", "staging_id": staging_id}
        elif args.review_command == "demote":
            analysis_id = await demote_staged_to_analysis(
                conn, args.staging_id, rationale=args.rationale
            )
            result = {"decision": "demoted", "analysis_id": analysis_id}
        else:  # pragma: no cover - argparse constrains this
            raise HmxPolicyError(
                f"unknown import review command: {args.review_command}"
            )
    finally:
        await conn.close()

    if args.json:
        sys.stdout.write(
            json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
        )
    else:
        from apps.cli_theme import console

        if args.review_command in (None, "list"):
            console.print(
                f"[heading]Pending HMX review[/heading]  [accent]{result['total']}[/accent] records"
            )
            for item in result["records"]:
                console.print(
                    f"  [key]{item['id']}[/key]  {item['section']}  "
                    f"[muted]{item.get('source_ref') or ''}[/muted]"
                )
        else:
            console.print(
                f"[ok]HMX review decision complete:[/ok] {result['decision']}"
            )
            for key in ("staging_id", "analysis_id", "local_ref"):
                if result.get(key):
                    console.print(f"  [key]{key}:[/key] {result[key]}")
    return 0
