"""Reusable local HTTP callback server for OAuth PKCE flows."""

from __future__ import annotations

import queue
import threading
import time
from http.server import BaseHTTPRequestHandler
from socketserver import TCPServer
from typing import Any
from urllib.parse import parse_qs, urlparse


def run_callback_server(
    port: int,
    callback_path: str = "/auth/callback",
    timeout_seconds: int = 60,
    expected_state: str | None = None,
    extract_params: list[str] | None = None,
    cancel_event: threading.Event | None = None,
    bind_host: str = "127.0.0.1",
) -> dict[str, str] | None:
    """Start a local HTTP server, wait for a callback, return extracted params.

    Returns a dict of extracted query params, or ``None`` on timeout or bind
    failure.  The server is always shut down before returning.

    ``bind_host`` widens the bind under WSL: the WSL2 localhost relay that
    forwards Windows-browser requests into the VM picks up wildcard binds
    far more reliably than 127.0.0.1-only ones. The handler still validates
    the callback path and OAuth state, and the server lives only for
    ``timeout_seconds``.
    """
    if extract_params is None:
        extract_params = ["code", "state"]

    result_queue: queue.Queue[dict[str, str]] = queue.Queue(maxsize=1)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: Any, **_kwargs: Any) -> None:  # noqa: ANN401
            return  # silence request logs

        def do_GET(self) -> None:  # noqa: N802
            try:
                parsed = urlparse(self.path or "")
                if parsed.path != callback_path:
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b"Not found")
                    return

                qs = parse_qs(parsed.query)

                if expected_state is not None:
                    req_state = (qs.get("state") or [""])[0]
                    if req_state != expected_state:
                        self.send_response(400)
                        self.end_headers()
                        self.wfile.write(b"State mismatch")
                        return

                extracted: dict[str, str] = {}
                for param in extract_params:  # type: ignore[union-attr]
                    val = (qs.get(param) or [""])[0]
                    if val:
                        extracted[param] = val

                if not extracted.get("code"):
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"Missing authorization code")
                    return

                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(
                    b"<!doctype html><html><body>"
                    b"<p>Authentication successful. Return to Hexis.</p>"
                    b"</body></html>"
                )
                try:
                    result_queue.put_nowait(extracted)
                except Exception:
                    pass
            except Exception:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"Internal error")

    server: TCPServer | None = None
    try:
        server = TCPServer((bind_host, port), Handler)
    except OSError:
        return None

    def _serve() -> None:
        assert server is not None
        with server:
            server.serve_forever(poll_interval=0.1)

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()

    result = None
    deadline = time.monotonic() + max(5, timeout_seconds)
    try:
        while time.monotonic() < deadline:
            if cancel_event and cancel_event.is_set():
                break
            try:
                result = result_queue.get(
                    timeout=min(0.2, max(0.01, deadline - time.monotonic()))
                )
                break
            except queue.Empty:
                continue
    finally:
        try:
            server.shutdown()
        except Exception:
            pass
        thread.join(timeout=1.0)

    return result
