"""Attached files are readable in the turn they are attached.

The composer's promise is simple: attach a PDF, ask about it, get an answer.
That rests on the fast read here — and, just as importantly, on it reporting
plainly when it produced nothing instead of failing quietly.
"""

from __future__ import annotations

import pytest

from services.attachments import (
    AttachmentText,
    attachment_kind,
    read_attachment_text,
)

_HAS_PDF = True
try:
    import pdfplumber  # noqa: F401
except ImportError:
    _HAS_PDF = False


def test_attachment_kind_uses_extension_then_mime():
    assert attachment_kind("Hartford.pdf", "application/pdf") == "document"
    assert attachment_kind("shot.png", "image/png") == "image"
    assert attachment_kind("note.m4a") == "audio"
    assert attachment_kind("clip.mov") == "video"
    # No extension to go on: the MIME type decides.
    assert attachment_kind("blob", "image/webp") == "image"
    assert attachment_kind("blob", "") == "document"


async def test_text_file_is_readable_immediately():
    result = await read_attachment_text(b"Terms of the agreement.", "terms.txt")

    assert result.readable
    assert result.text == "Terms of the agreement."
    assert result.truncated is False
    assert result.reason is None


async def test_long_text_is_truncated_and_says_so():
    result = await read_attachment_text(b"x" * 5000, "long.txt", max_chars=100)

    assert result.truncated is True
    assert len(result.text) == 100
    assert result.chars == 5000


async def test_images_and_media_are_left_to_their_own_paths():
    image = await read_attachment_text(b"\x89PNG...", "shot.png")
    audio = await read_attachment_text(b"ID3...", "call.mp3")

    # Not an error — the image rides the turn visually and audio needs a
    # transcript, so neither belongs in a synchronous text read.
    assert image.reason == "image"
    assert image.text == ""
    assert audio.reason == "audio"


async def test_oversized_files_are_deferred_not_attempted():
    result = await read_attachment_text(b"x" * 2048, "big.txt", max_bytes=1024)

    assert result.reason == "too_large"
    assert result.readable is False


async def test_unsupported_format_reports_instead_of_raising():
    # Legacy .xls fails loud inside the reader; the composer must survive it
    # with a stated reason rather than a traceback.
    result = await read_attachment_text(b"\xd0\xcf\x11\xe0", "legacy.xls")

    assert isinstance(result, AttachmentText)
    assert result.reason == "error"
    assert result.error
    assert result.readable is False


@pytest.mark.skipif(not _HAS_PDF, reason="pdfplumber not installed")
async def test_pdf_text_is_available_in_the_same_turn():
    # A minimal one-page text PDF written by hand (no reportlab dependency).
    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
        b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
        b"4 0 obj<</Length 64>>stream\n"
        b"BT /F1 12 Tf 72 720 Td (Manning grants the Author a royalty.) Tj ET\n"
        b"endstream endobj\n"
        b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"trailer<</Root 1 0 R>>\n"
        b"%%EOF\n"
    )

    result = await read_attachment_text(pdf_bytes, "Hartford.pdf")

    assert result.readable
    assert "Manning grants the Author a royalty." in result.text
    assert result.extractor == "pdfplumber"
