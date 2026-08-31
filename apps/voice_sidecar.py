"""Loopback-only entrypoint for Piper's official HTTP synthesis server."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hexis local voice sidecar")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=42667)
    parser.add_argument("--model", required=True)
    # A random launch marker lets the CLI distinguish the exact process it
    # created from a reused PID. It is intentionally not forwarded to Piper.
    parser.add_argument("--hexis-owner-token", default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        sys.stderr.write(
            "Hexis voice refuses a non-loopback bind. Keep the sidecar on "
            "127.0.0.1 and use the dashboard's private HTTPS route.\n"
        )
        return 2
    if not 1 <= args.port <= 65_535:
        sys.stderr.write("Voice sidecar port must be between 1 and 65535.\n")
        return 2
    try:
        from piper.http_server import main as piper_http_main
    except ImportError:
        sys.stderr.write(
            "Piper voice support is not installed in this Hexis environment. "
            "Run `hexis voice setup` to install it and continue in place.\n"
        )
        return 2

    previous = sys.argv
    sys.argv = [
        "piper.http_server",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model",
        args.model,
    ]
    try:
        piper_http_main()
    finally:
        sys.argv = previous
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
