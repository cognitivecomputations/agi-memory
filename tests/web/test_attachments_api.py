"""Chat attachments: preserved and read on attach, ingested only on send.

The composer's promise ("attach a PDF, ask about it, get an answer") lives on
these two endpoints. The split matters as much as the read: a file the user
attaches and then removes must leave nothing behind in the agent's memory.
"""

from __future__ import annotations

import pytest

import apps.hexis_api as web_module
from apps.hexis_api import app

pytestmark = [pytest.mark.asyncio(loop_scope="session")]

# A minimal one-page text PDF written by hand (no reportlab dependency).
PDF_BYTES = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
    b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
    b"4 0 obj<</Length 78>>stream\n"
    b"BT /F1 12 Tf 72 720 Td (Manning grants the Author a royalty on net receipts.) Tj ET\n"
    b"endstream endobj\n"
    b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
    b"trailer<</Root 1 0 R>>\n"
    b"%%EOF\n"
)


@pytest.fixture
async def client(db_pool):
    import httpx

    original_pool = web_module._pool
    web_module._pool = db_pool
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client
    web_module._pool = original_pool


async def _cleanup(db_pool, sha256: str | None) -> None:
    if not sha256:
        return
    async with db_pool.acquire() as conn:
        await conn.execute("DELETE FROM ingestion_jobs WHERE content_hash = $1", f"artifact:{sha256}")
        await conn.execute("DELETE FROM source_artifacts WHERE sha256 = $1", sha256)


async def test_attaching_a_pdf_returns_its_text_without_ingesting_it(client, db_pool):
    sha = None
    try:
        resp = await client.post(
            "/api/attachments",
            files={"file": ("Hartford.pdf", PDF_BYTES, "application/pdf")},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        sha = body["sha256"]

        # The text is in hand right away — that is the whole point.
        assert body["readable"] is True
        assert "Manning grants the Author a royalty" in body["text"]
        assert body["kind"] == "document"
        assert body["truncated"] is False

        async with db_pool.acquire() as conn:
            artifact = await conn.fetchrow(
                "SELECT original_filename FROM source_artifacts WHERE sha256 = $1", sha
            )
            queued = await conn.fetchval(
                "SELECT count(*) FROM ingestion_jobs WHERE content_hash = $1", f"artifact:{sha}"
            )
        # Preserved, but nothing was committed to memory: the user has not sent yet.
        assert artifact["original_filename"] == "Hartford.pdf"
        assert queued == 0
    finally:
        await _cleanup(db_pool, sha)


async def test_sending_the_message_is_what_files_the_attachment(client, db_pool):
    sha = None
    try:
        prepared = (
            await client.post(
                "/api/attachments",
                files={"file": ("terms.txt", b"Retention window is 90 days.", "text/plain")},
            )
        ).json()
        sha = prepared["sha256"]
        assert prepared["text"] == "Retention window is 90 days."

        resp = await client.post(
            f"/api/attachments/{prepared['artifact_id']}/ingest",
            json={"filename": "terms.txt", "mode": "fast"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["accepted"] is True

        async with db_pool.acquire() as conn:
            job = await conn.fetchrow(
                "SELECT kind, payload FROM ingestion_jobs WHERE content_hash = $1",
                f"artifact:{sha}",
            )
        assert job is not None
        assert job["kind"] == "artifact"
    finally:
        await _cleanup(db_pool, sha)


async def test_unknown_artifact_says_what_to_do_next(client):
    resp = await client.post(
        "/api/attachments/11111111-1111-4111-8111-111111111111/ingest", json={}
    )
    assert resp.status_code == 404
    assert "re-attach" in resp.json()["detail"]

    resp = await client.post("/api/attachments/not-a-uuid/ingest", json={})
    assert resp.status_code == 422


async def test_empty_upload_is_refused(client):
    resp = await client.post("/api/attachments", files={"file": ("empty.txt", b"", "text/plain")})
    assert resp.status_code == 422
