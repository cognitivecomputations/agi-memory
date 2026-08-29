"""CLI for validating, running, scoring, and comparing memory benchmark v1.

Examples:
    python -m evals.memory_benchmark.run validate
    python -m evals.memory_benchmark.run run --adapter hexis
    python -m evals.memory_benchmark.run run --adapter command \
        --name my-agent --command './my-wrapper --json'
    python -m evals.memory_benchmark.run compare result-a.json result-b.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

from core.agent_api import db_dsn_from_env

from .adapters import (
    AppendOnlyTranscriptAdapter,
    CommandAdapter,
    HexisMemoryAdapter,
    MemoryAdapter,
    RecentWindowAdapter,
)
from .model import (
    BENCHMARK_NAME,
    BENCHMARK_VERSION,
    BenchmarkCase,
    Prediction,
    dataset_path,
    dataset_sha256,
    load_cases,
)
from .scoring import score_predictions


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _version_state() -> dict[str, str | None]:
    source_version: str | None = None
    source_root = Path(__file__).resolve().parents[2]
    project = source_root / "pyproject.toml"
    try:
        match = re.search(
            r"(?ms)^\[project\]\s*$.*?^version\s*=\s*[\"']([^\"']+)[\"']",
            project.read_text(encoding="utf-8"),
        )
        source_version = match.group(1) if match else None
    except OSError:
        pass
    installed_version: str | None = None
    candidates = list(distributions(name="hexis"))
    if project.exists():
        # A setuptools build leaves ``hexis.egg-info`` in the checkout. That
        # metadata shadows the actual environment distribution on sys.path, so
        # exclude distributions rooted at the source checkout when reporting the
        # installed version.
        for candidate in candidates:
            distribution_root = Path(candidate.locate_file("")).resolve()
            if distribution_root != source_root:
                installed_version = candidate.version
                break
    elif candidates:
        installed_version = candidates[0].version
    return {
        "hexis_source_version": source_version,
        "installed_distribution_version": installed_version,
    }


def _source_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
        )
        return {"git_revision": revision, "git_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"git_revision": None, "git_dirty": None}


def _cache_dir() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
    return base / "hexis" / "memory-benchmark"


def _default_output(adapter_name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = "-".join(re.findall(r"[a-z0-9]+", adapter_name.casefold())) or "run"
    return _cache_dir() / f"{safe_name}-{stamp}.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _load_prediction_document(path: Path) -> list[Prediction]:
    text = path.read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(raw, dict):
        raw = raw.get("predictions")
    if not isinstance(raw, list):
        raise ValueError("prediction input must be a JSON array, JSONL, or run result")
    return [Prediction.from_dict(item) for item in raw if isinstance(item, dict)]


async def _run_adapter(
    adapter: MemoryAdapter,
    cases: list[BenchmarkCase],
    *,
    dataset_file: Path | None = None,
) -> dict[str, Any]:
    source = dataset_file or dataset_path()
    root = Path(__file__).resolve().parents[2]
    try:
        dataset_label = str(source.resolve().relative_to(root))
    except ValueError:
        dataset_label = str(source.resolve())
    started_at = _now()
    predictions: list[Prediction] = []
    timings: list[dict[str, Any]] = []
    try:
        for index, case in enumerate(cases, 1):
            start = time.perf_counter()
            prediction = await adapter.predict(case)
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            predictions.append(prediction)
            timings.append({"case_id": case.case_id, "latency_ms": latency_ms})
            print(
                f"[{index:02d}/{len(cases):02d}] {case.case_id}: "
                f"{'ABSTAIN' if prediction.abstained else 'answered'} ({latency_ms}ms)",
                file=sys.stderr,
            )
    finally:
        await adapter.close()
    score = score_predictions(cases, predictions)
    return {
        "format_version": "1.0",
        "benchmark": {
            "name": BENCHMARK_NAME,
            "version": BENCHMARK_VERSION,
            "dataset": dataset_label,
            "dataset_sha256": dataset_sha256(source),
            "custom_dataset": dataset_file is not None,
            "case_count": len(cases),
        },
        "system": {
            **adapter.run_metadata(),
            **_version_state(),
            **_source_state(),
        },
        "execution": {
            "started_at": started_at,
            "completed_at": _now(),
            "case_timings": timings,
            "total_latency_ms": round(sum(item["latency_ms"] for item in timings), 2),
        },
        "predictions": [prediction.as_dict() for prediction in predictions],
        "score": score,
    }


async def _build_adapter(args: argparse.Namespace) -> MemoryAdapter:
    if args.adapter == "append-only":
        return AppendOnlyTranscriptAdapter()
    if args.adapter == "recent-window":
        return RecentWindowAdapter()
    if args.adapter == "command":
        if not args.command:
            raise ValueError("--adapter command requires --command")
        return CommandAdapter(
            args.command,
            name=args.name or "external-command",
            timeout_seconds=args.timeout,
        )
    if args.adapter == "hexis":
        import asyncpg

        pool = await asyncpg.create_pool(
            args.dsn or db_dsn_from_env(),
            min_size=1,
            max_size=1,
            timeout=10,
            command_timeout=180,
        )
        adapter = HexisMemoryAdapter(
            pool,
            live_contradictions=args.live_contradictions,
        )

        async def close() -> None:
            await pool.close()

        adapter.close = close  # type: ignore[method-assign]
        return adapter
    raise ValueError(f"unknown adapter: {args.adapter}")


def _print_summary(result: dict[str, Any], output: Path) -> None:
    score = result["score"]
    print(f"\n{result['system']['name']}: {score['overall']:.2f}")
    for dimension, detail in score["dimensions"].items():
        print(f"  {dimension:<28} {detail['score']:>6.2f}  ({detail['cases']} cases)")
    print(f"result: {output}")


async def _run_command(args: argparse.Namespace) -> int:
    dataset_file = Path(args.dataset) if args.dataset else None
    cases = load_cases(dataset_file)
    adapter = await _build_adapter(args)
    result = await _run_adapter(adapter, cases, dataset_file=dataset_file)
    output = (
        Path(args.output).expanduser() if args.output else _default_output(adapter.name)
    )
    _write_json(output, result)
    _print_summary(result, output)
    return 0


async def _run_all_command(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else _cache_dir()
    dataset_file = Path(args.dataset) if args.dataset else None
    cases = load_cases(dataset_file)
    adapters = ["append-only", "recent-window", "hexis"]
    results: list[dict[str, Any]] = []
    for adapter_name in adapters:
        local_args = argparse.Namespace(**vars(args))
        local_args.adapter = adapter_name
        local_args.command = None
        local_args.name = None
        adapter = await _build_adapter(local_args)
        result = await _run_adapter(adapter, cases, dataset_file=dataset_file)
        target = output_dir / f"{result['system']['name']}.json"
        _write_json(target, result)
        _print_summary(result, target)
        results.append(result)
    comparison = _comparison(results)
    target = output_dir / "comparison.json"
    _write_json(target, comparison)
    print(f"comparison: {target}")
    return 0


def _comparison(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for result in results:
        score = result.get("score") or {}
        rows.append(
            {
                "name": (result.get("system") or {}).get("name"),
                "kind": (result.get("system") or {}).get("kind"),
                "overall": score.get("overall"),
                "dimensions": {
                    name: detail.get("score")
                    for name, detail in (score.get("dimensions") or {}).items()
                },
                "dataset_sha256": (result.get("benchmark") or {}).get("dataset_sha256"),
            }
        )
    hashes = {row["dataset_sha256"] for row in rows}
    if len(hashes) > 1:
        raise ValueError("cannot compare results produced from different datasets")
    return {
        "benchmark": BENCHMARK_NAME,
        "version": BENCHMARK_VERSION,
        "dataset_sha256": next(iter(hashes), dataset_sha256()),
        "generated_at": _now(),
        "results": rows,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hexis Public Memory Benchmark v1")
    commands = parser.add_subparsers(dest="command_name", required=True)

    validate = commands.add_parser("validate", help="Validate the public corpus")
    validate.add_argument("--dataset")

    run = commands.add_parser("run", help="Run one adapter and write a scored result")
    run.add_argument(
        "--adapter",
        choices=["hexis", "append-only", "recent-window", "command"],
        required=True,
    )
    run.add_argument("--dataset")
    run.add_argument("--output")
    run.add_argument("--dsn", help=argparse.SUPPRESS)
    run.add_argument("--live-contradictions", action="store_true")
    run.add_argument("--command", help="External adapter argv; parsed without a shell")
    run.add_argument("--name", help="Published external system name")
    run.add_argument("--timeout", type=int, default=180)

    run_all = commands.add_parser(
        "run-all", help="Run Hexis and both reference baselines"
    )
    run_all.add_argument("--output-dir")
    run_all.add_argument("--dsn", help=argparse.SUPPRESS)
    run_all.add_argument("--live-contradictions", action="store_true")
    run_all.add_argument("--dataset")
    run_all.add_argument("--timeout", type=int, default=180)

    score = commands.add_parser("score", help="Score a predictions file")
    score.add_argument("predictions")
    score.add_argument("--dataset")
    score.add_argument("--output")

    compare = commands.add_parser("compare", help="Compare scored run result files")
    compare.add_argument("results", nargs="+")
    compare.add_argument("--output")

    show = commands.add_parser("show", help="Print one gold-free adapter input case")
    show.add_argument("case_id")
    show.add_argument("--dataset")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command_name == "validate":
            cases = load_cases(Path(args.dataset) if args.dataset else None)
            print(
                json.dumps(
                    {
                        "benchmark": BENCHMARK_NAME,
                        "version": BENCHMARK_VERSION,
                        "cases": len(cases),
                        "dataset_sha256": dataset_sha256(
                            Path(args.dataset) if args.dataset else None
                        ),
                    },
                    indent=2,
                )
            )
            return 0
        if args.command_name == "run":
            return asyncio.run(_run_command(args))
        if args.command_name == "run-all":
            return asyncio.run(_run_all_command(args))
        if args.command_name == "score":
            cases = load_cases(Path(args.dataset) if args.dataset else None)
            result = score_predictions(
                cases, _load_prediction_document(Path(args.predictions))
            )
            if args.output:
                _write_json(Path(args.output), result)
            else:
                print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if args.command_name == "compare":
            documents = [
                json.loads(Path(path).read_text(encoding="utf-8"))
                for path in args.results
            ]
            result = _comparison(documents)
            if args.output:
                _write_json(Path(args.output), result)
            else:
                print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if args.command_name == "show":
            cases = load_cases(Path(args.dataset) if args.dataset else None)
            case = next((item for item in cases if item.case_id == args.case_id), None)
            if case is None:
                raise ValueError(f"unknown case_id: {args.case_id}")
            print(json.dumps(case.public_dict(), indent=2, ensure_ascii=False))
            return 0
    except (OSError, ValueError) as exc:
        print(f"memory benchmark: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
