#!/usr/bin/env python3
"""Tiny deterministic fake embedding server for CI (no GB-scale model).

It speaks the local embed ``/api/embed`` shape that Hexis's ``get_embedding()`` expects
(``db/03_functions_helpers.sql``): a health probe on ``GET /api/tags`` and
``POST /api/embed`` returning ``{"embeddings": [[...768 floats...], ...]}``.

Each input text maps to a **stable** 768-float vector derived from its SHA-256, so
different content yields different vectors (required by
``test_get_embedding_different_content_different_embeddings``) while identical
content is reproducible. The vectors carry **no semantics** — tests that need
similarity ordering inject their own ``array_fill(...)`` vectors instead.

Run: ``EMBEDDING_DIMENSION=768 FAKE_EMBEDDINGS_PORT=11435 python ops/ci/fake_embeddings.py``
Point the DB at it via GUC ``app.embedding_service_url=http://<host>:<port>/api/embed``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DIM = int(os.getenv("EMBEDDING_DIMENSION", "768"))

# ``get_embedding()`` prefixes documents with "search_document: " while the
# recall ranker prefixes queries with "search_query: " (ensure_embedding_prefix).
# A real embedding model treats those task prefixes as near-transparent — the
# same text embeds to nearly the same point in either role. Hash the RAW text
# so this fake behaves the same way; otherwise a stored memory and a recall
# query for the identical text would get unrelated vectors.
_TASK_PREFIX = re.compile(
    r"^\s*(search_document|search_query|clustering|classification)\s*:\s*",
    re.IGNORECASE,
)


def _vector(text: str) -> list[float]:
    """A deterministic pseudo-random positive vector in [0, 1], seeded by text.

    Keeping values positive matters for DB tests that compare recall scores
    against hand-authored ``array_fill(positive, ...)`` vectors: those tests are
    about strength/fidelity ordering, not semantic embedding polarity.
    """
    text = _TASK_PREFIX.sub("", text)
    out: list[float] = []
    counter = 0
    while len(out) < DIM:
        digest = hashlib.sha256(f"{text}\x00{counter}".encode("utf-8")).digest()
        for i in range(0, len(digest), 4):
            (u,) = struct.unpack("<I", digest[i:i + 4])
            out.append(u / 2**32)
            if len(out) >= DIM:
                break
        counter += 1
    return out[:DIM]


class _Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 (http.server API)
        # /api/tags (tag-style health) and /health (origin health) both return 200.
        self._json(200, {"models": []})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            req = json.loads(raw or b"{}")
        except Exception:
            req = {}
        inputs = req.get("input") or req.get("inputs") or req.get("prompt") or []
        if isinstance(inputs, str):
            inputs = [inputs]
        vectors = [_vector(str(t)) for t in inputs]
        # Local embed shape; get_embedding() also accepts {data:[{embedding}]} / {embedding}.
        self._json(200, {"embeddings": vectors})

    def log_message(self, *args):  # keep CI logs quiet
        pass


def main() -> None:
    host = os.getenv("FAKE_EMBEDDINGS_HOST", "0.0.0.0")
    port = int(os.getenv("FAKE_EMBEDDINGS_PORT", "11434"))
    server = ThreadingHTTPServer((host, port), _Handler)
    print(f"fake-embeddings listening on {host}:{port} (dim={DIM})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
