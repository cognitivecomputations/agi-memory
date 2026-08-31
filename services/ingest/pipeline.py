"""Hexis ingestion — split from the former services/ingest.py (#89).
Module: pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import mimetypes
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import UUID

from core.integration_reliability import (
    IntegrationHttpError,
    format_provider_error,
    request_bytes_response,
)

from .config import (
    Appraisal,
    Config,
    DocumentInfo,
    Extraction,
    IngestionMetrics,
    IngestionMode,
    Section,
    _emit,
    _extract_title,
    _hash_text,
    _infer_source_type,
    _normalize_mode,
    _should_cancel,
    _word_count,
)
from .llm import Appraiser, IngestLLM, KnowledgeExtractor
from .readers import (
    AudioReader,
    CodeReader,
    DataReader,
    DocxReader,
    EmailReader,
    EpubReader,
    ImageReader,
    LatexReader,
    NotebookReader,
    PDFReader,
    PptxReader,
    ReaderResult,
    RssReader,
    RtfReader,
    VideoReader,
    WebReader,
    XlsxReader,
    YouTubeTranscriptReader,
    get_reader,
)
from .sectioning import CHUNKER_VERSION, Sectioner
from .store import MemoryStore

# =========================================================================
# INGESTION PIPELINE
# =========================================================================


class IngestionPipeline:
    SUPPORTED_EXTENSIONS = (
        # Documents
        {".md", ".markdown", ".txt", ".text", ".pdf"}
        | DocxReader.EXTENSIONS
        | RtfReader.EXTENSIONS
        | LatexReader.EXTENSIONS
        | EmailReader.EXTENSIONS
        | EpubReader.EXTENSIONS
        | PptxReader.EXTENSIONS
        | XlsxReader.EXTENSIONS
        | NotebookReader.EXTENSIONS
        # Code
        | set(CodeReader.LANGUAGE_MAP.keys())
        # Data
        | DataReader.DATA_EXTENSIONS
        # Media
        | ImageReader.IMAGE_EXTENSIONS
        | AudioReader.AUDIO_EXTENSIONS
        | VideoReader.VIDEO_EXTENSIONS
    )

    def __init__(self, config: Config):
        self.config = config
        self.config.mode = _normalize_mode(self.config.mode)
        self.sectioner = Sectioner(config.max_section_chars, config.chunk_overlap)
        self.llm = IngestLLM(config)
        self.appraiser = Appraiser(self.llm)
        self.extractor = KnowledgeExtractor(self.llm)
        self.store = MemoryStore(config)
        self.stats = {"files_processed": 0, "memories_created": 0, "errors": 0}
        self.last_result: dict[str, Any] = {}
        # DB-owned spreadsheet row cap; capping always warns (readers.py).
        XlsxReader.MAX_ROWS_PER_SHEET = max(1, int(config.xlsx_max_rows_per_sheet or 5000))

    def _detect_url_source_type(self, url: str) -> str:
        """Route URL ingestion to the reader that will preserve useful shape."""
        if YouTubeTranscriptReader.can_handle(url):
            return "youtube"

        from urllib.parse import urlparse

        path_lower = urlparse(url).path.lower()
        if path_lower.endswith(".pdf") or path_lower.startswith("/pdf/"):
            return "pdf"
        if path_lower.endswith((".rss", ".atom", ".xml")):
            return "rss"
        return "web"

    async def _read_pdf_url_result(self, url: str) -> tuple[ReaderResult, dict[str, Any]]:
        import tempfile
        from urllib.parse import urlparse

        try:
            response = await request_bytes_response(
                "url_ingest",
                "GET",
                url,
                headers={"User-Agent": "Hexis/1.0"},
                timeout=60.0,
                attempts=3,
                max_delay=15.0,
                follow_redirects=True,
                max_bytes=self.config.upload_max_bytes,
            )
        except IntegrationHttpError as exc:
            raise RuntimeError(format_provider_error("URL ingest", exc)) from exc

        raw = response.content
        mime_type = response.headers.get("content-type") or "application/pdf"

        original_filename = Path(urlparse(url).path).name or "download.pdf"
        if not original_filename.lower().endswith(".pdf"):
            original_filename = f"{original_filename}.pdf"

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(raw)
            tmp.flush()
            reader_result = PDFReader.read_result(Path(tmp.name))

        header = f"[Source: {url}]\n[Format: PDF]\n\n"
        reader_result.text = header + reader_result.text
        for anchor in reader_result.anchors:
            anchor.char_offset += len(header)
        reader_result.metadata.update({"url": url})

        artifact_info = self._prepare_artifact(
            raw,
            original_filename=original_filename,
            mime_type=mime_type,
            storage_kind="url",
            metadata={
                "url": url,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        artifact_info["storage_ref"] = artifact_info.get("storage_ref") or url
        return reader_result, artifact_info

    async def ingest_file(self, file_path: Path) -> int:
        self.last_result = {}
        metrics = IngestionMetrics(start_time=time.time())
        if _should_cancel(self.config):
            raise RuntimeError("Ingestion cancelled")
        if not file_path.exists():
            _emit(self.config, f"File not found: {file_path}")
            return 0
        if file_path.suffix.lower() == ".xls":
            _emit(self.config, "Legacy .xls is not supported; convert it to .xlsx "
                               "(e.g. soffice --convert-to xlsx) and re-ingest.")
            return 0
        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            _emit(self.config, f"Unsupported file type: {file_path.suffix}")
            return 0
        if self.config.verbose:
            _emit(self.config, f"\nProcessing: {file_path}")

        # Preserve the original bytes BEFORE extraction: a failed parse never
        # loses the source, and a better extractor can re-run later.
        artifact_info = None
        try:
            artifact_info = self._prepare_artifact(
                file_path.read_bytes(),
                original_filename=file_path.name,
                mime_type=mimetypes.guess_type(str(file_path))[0],
            )
        except Exception as exc:
            _emit(self.config, f"  Warning: could not capture original artifact: {exc}")

        reader = get_reader(file_path)
        try:
            reader_result = reader.read_result(file_path)
            content = reader_result.text
            metrics.source_size_bytes = len(content.encode("utf-8"))
        except Exception as exc:
            _emit(self.config, f"  Error reading file: {exc}")
            self.stats["errors"] += 1
            metrics.errors.append(str(exc))
            await self._preserve_failed_extraction(reader, artifact_info, exc, file_path)
            return 0
        for warning in reader_result.warnings:
            _emit(self.config, f"  Extraction warning [{warning.code}]: {warning.message}")

        doc = DocumentInfo(
            title=_extract_title(content, file_path),
            source_type=_infer_source_type(file_path),
            content_hash=_hash_text(content),
            word_count=_word_count(content),
            path=str(file_path),
            file_type=file_path.suffix.lower(),
        )
        return await self._ingest_content(
            content, doc, metrics, section_path=file_path,
            reader_result=reader_result, artifact_info=artifact_info,
        )

    def _prepare_artifact(
        self,
        raw: bytes,
        *,
        original_filename: str | None,
        mime_type: str | None,
        storage_kind: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Decide where the original bytes live: in-DB up to the configured
        threshold (rides pg_dump backups), else a content-addressed file in
        the managed artifact directory."""
        from .artifacts import prepare_artifact_info

        return prepare_artifact_info(
            raw,
            original_filename=original_filename,
            mime_type=mime_type,
            storage_kind=storage_kind,
            metadata=metadata,
            max_db_bytes=self.config.artifact_max_db_bytes,
            artifact_dir=self.config.resolve_artifact_dir(),
        )

    async def _preserve_failed_extraction(
        self,
        reader: Any,
        artifact_info: dict[str, Any] | None,
        exc: Exception,
        file_path: Path,
    ) -> None:
        """Reader failure: the artifact is still preserved (document-less)
        and the failed run recorded, so the user can see what failed and why
        — and a re-ingest after fixing deps re-links by sha256."""
        if artifact_info is None:
            return
        try:
            artifact = await self.store.upsert_source_artifact(
                sha256=artifact_info["sha256"],
                storage_kind=artifact_info["storage_kind"],
                data=artifact_info.get("bytes"),
                storage_ref=artifact_info.get("storage_ref"),
                source_document_id=None,
                original_filename=artifact_info.get("original_filename"),
                mime_type=artifact_info.get("mime_type"),
                byte_size=artifact_info.get("byte_size"),
                metadata=artifact_info.get("metadata"),
            )
            await self.store.record_extraction_run(
                document_id=None,
                artifact_id=artifact.get("artifact_id"),
                extractor_name=type(reader).__name__,
                status="failed",
                errors=[{"message": str(exc), "path": str(file_path)}],
            )
            _emit(self.config,
                  f"  Original artifact preserved (sha256={artifact_info['sha256'][:12]}...); "
                  "fix the reader/dependencies and re-ingest to retry.")
        except Exception as exc2:
            _emit(self.config, f"  Warning: could not preserve failed-extraction artifact: {exc2}")

    GIT_IGNORE_DIRS = {
        ".git", "node_modules", "__pycache__", ".venv", "venv",
        ".env", "dist", "build", ".tox", ".mypy_cache",
        ".pytest_cache", "vendor", ".bundle", ".next", "coverage",
    }

    async def ingest_directory(
        self,
        dir_path: Path,
        recursive: bool = True,
        exclude_dirs: set[str] | None = None,
    ) -> int:
        self.last_result = {}
        if _should_cancel(self.config):
            raise RuntimeError("Ingestion cancelled")
        if not dir_path.exists() or not dir_path.is_dir():
            _emit(self.config, f"Directory not found: {dir_path}")
            return 0
        pattern = "**/*" if recursive else "*"
        files = [
            f for f in dir_path.glob(pattern)
            if f.is_file()
            and f.suffix.lower() in self.SUPPORTED_EXTENSIONS
            and (
                exclude_dirs is None
                or not any(part in exclude_dirs for part in f.relative_to(dir_path).parts)
            )
        ]
        if self.config.verbose:
            _emit(self.config, f"Found {len(files)} files to process")

        # Bounded file concurrency (#90); per-file work is already parallel
        # inside, so this multiplies modestly, not explosively.
        file_slots = asyncio.Semaphore(max(1, self.config.max_parallel_files))

        async def _one(file_path: Path) -> int:
            async with file_slots:
                return await self.ingest_file(file_path)

        previous_auto_load = self.config.auto_load_to_desk
        self.config.auto_load_to_desk = False
        try:
            counts = await asyncio.gather(*[_one(f) for f in files])
        finally:
            self.config.auto_load_to_desk = previous_auto_load
        total = sum(counts)
        self.last_result = {
            "memories_created": total,
            "bulk_ingest": True,
            "files_processed": len(files),
            "desk_load": {
                "skipped": True,
                "reason": "bulk_directory_ingest",
            },
        }
        return total

    async def ingest_url(self, url: str, title: str | None = None) -> int:
        """Ingest content from a URL."""
        self.last_result = {}
        metrics = IngestionMetrics(start_time=time.time())

        if self.config.verbose:
            _emit(self.config, f"\nFetching: {url}")

        source_type = self._detect_url_source_type(url)
        reader_result: ReaderResult | None = None
        artifact_info: dict[str, Any] | None = None
        try:
            if source_type == "youtube":
                content = YouTubeTranscriptReader.read(url)
                reader_result = ReaderResult(
                    text=content,
                    extractor_name="YouTubeTranscriptReader",
                    metadata={"url": url},
                )
            elif source_type == "pdf":
                reader_result, artifact_info = await self._read_pdf_url_result(url)
                content = reader_result.text
            elif source_type == "rss":
                content = RssReader.read(url)
                if not content:
                    raise RuntimeError("No RSS entries found")
                reader_result = ReaderResult(
                    text=content,
                    extractor_name="RssReader",
                    metadata={"url": url},
                )
            else:
                reader_result = WebReader.read_result(url)
                content = reader_result.text
                if reader_result.artifact_bytes:
                    # The fetched HTML is the original artifact for a web source.
                    try:
                        artifact_info = self._prepare_artifact(
                            reader_result.artifact_bytes,
                            original_filename=None,
                            mime_type=reader_result.artifact_mime or "text/html",
                            storage_kind="url",
                            metadata={
                                "url": url,
                                "fetched_at": reader_result.metadata.get("fetched_at"),
                            },
                        )
                        artifact_info["storage_ref"] = artifact_info.get("storage_ref") or url
                    except Exception as exc:
                        _emit(self.config, f"  Warning: could not capture web artifact: {exc}")
            metrics.source_size_bytes = len(content.encode("utf-8"))
        except Exception:
            if source_type == "web":
                # Fallback: try as RSS/Atom feed
                try:
                    content = RssReader.read(url)
                    if content:
                        source_type = "rss"
                        reader_result = ReaderResult(
                            text=content,
                            extractor_name="RssReader",
                            metadata={"url": url},
                        )
                        metrics.source_size_bytes = len(content.encode("utf-8"))
                        if self.config.verbose:
                            _emit(self.config, "  Fetched as RSS/Atom feed")
                    else:
                        raise RuntimeError("No RSS entries found")
                except Exception as exc2:
                    _emit(self.config, f"  Error fetching URL: {exc2}")
                    self.stats["errors"] += 1
                    metrics.errors.append(str(exc2))
                    return 0
            else:
                raise

        if not title:
            title_match = re.search(r"\[Title: (.+?)\]", content)
            if title_match:
                title = title_match.group(1)
            else:
                title = url.split("/")[-1] or url

        doc = DocumentInfo(
            title=title,
            source_type=source_type,
            content_hash=_hash_text(content),
            word_count=_word_count(content),
            path=url,
            file_type=".html" if source_type == "web" else f".{source_type}",
        )
        return await self._ingest_content(
            content, doc, metrics, section_path=Path("web_content.md"),
            reader_result=reader_result, artifact_info=artifact_info,
        )

    async def ingest_text(
        self,
        content: str,
        *,
        title: str | None = None,
        source_type: str = "pasted_text",
        path: str | None = None,
        file_type: str = ".md",
    ) -> int:
        """Ingest raw text (pasted documents, job payloads) — no file needed."""
        self.last_result = {}
        metrics = IngestionMetrics(start_time=time.time())
        metrics.source_size_bytes = len(content.encode("utf-8"))
        if not title:
            first_line = next((ln.strip() for ln in content.splitlines() if ln.strip()), "")
            title = first_line[:80] or "Pasted text"

        doc = DocumentInfo(
            title=title,
            source_type=source_type,
            content_hash=_hash_text(content),
            word_count=_word_count(content),
            path=path or f"text:{_hash_text(content)[:12]}",
            file_type=file_type,
        )
        return await self._ingest_content(
            content, doc, metrics, section_path=Path("pasted_text.md")
        )

    async def _ingest_content(
        self,
        content: str,
        doc: DocumentInfo,
        metrics: IngestionMetrics,
        *,
        section_path: Path,
        reader_result: ReaderResult | None = None,
        artifact_info: dict[str, Any] | None = None,
    ) -> int:
        """The single ingestion core (#89): every entry point — file, URL,
        raw text, stdin — reduces to a reader over this."""
        llm_calls_start = self.llm.call_count
        mode = self.config.mode
        metrics.word_count = doc.word_count
        metrics.mode = mode.value
        metrics.source_type = doc.source_type

        stored_doc = await self.store.store_source_document(
            title=doc.title,
            source_type=doc.source_type,
            content_hash=doc.content_hash,
            path=doc.path,
            file_type=doc.file_type,
            content=content,
            word_count=doc.word_count,
            source_attribution=self._source_payload(doc),
            metadata={"mode": mode.value},
        )
        doc.document_id = str(stored_doc.get("document_id") or "") or None

        # Artifact link + extraction receipt: warnings become a durable,
        # inspectable record instead of vanishing into logs. Failures here
        # never block memory extraction — warn loud, continue.
        if doc.document_id:
            artifact_id = None
            try:
                if artifact_info is not None:
                    artifact = await self.store.upsert_source_artifact(
                        sha256=artifact_info["sha256"],
                        storage_kind=artifact_info["storage_kind"],
                        data=artifact_info.get("bytes"),
                        storage_ref=artifact_info.get("storage_ref"),
                        source_document_id=doc.document_id,
                        original_filename=artifact_info.get("original_filename"),
                        mime_type=artifact_info.get("mime_type"),
                        byte_size=artifact_info.get("byte_size"),
                        metadata=artifact_info.get("metadata"),
                    )
                    artifact_id = artifact.get("artifact_id")
                else:
                    # No separate artifact: the normalized content IS the source.
                    await self.store.set_original_hash(doc.document_id, doc.content_hash)
            except Exception as exc:
                _emit(self.config, f"  Warning: artifact preservation failed: {exc}")
            if reader_result is not None:
                try:
                    await self.store.record_extraction_run(
                        document_id=doc.document_id,
                        artifact_id=artifact_id,
                        extractor_name=reader_result.extractor_name or "unknown",
                        extractor_version=reader_result.extractor_version,
                        status="completed",
                        warnings=reader_result.warnings_payload(),
                        metadata=reader_result.metadata,
                    )
                except Exception as exc:
                    _emit(self.config, f"  Warning: extraction-run record failed: {exc}")

        sections = self.sectioner.split(
            content, section_path,
            anchors=reader_result.anchors if reader_result and reader_result.anchors else None,
        )
        section_hashes = [_hash_text(s.content) for s in sections]

        # Durable chunks land before the receipt gate so documents ingested
        # under older versions still gain chunks on a re-run. Chunk failure
        # never blocks memory extraction — warn loud, continue.
        if doc.document_id:
            try:
                chunk_result = await self.store.upsert_source_document_chunks(
                    doc.document_id, sections, CHUNKER_VERSION
                )
                doc.chunk_ids = {
                    i: str(cid)
                    for i, cid in enumerate(chunk_result.get("chunk_ids") or [])
                }
            except Exception as exc:
                _emit(self.config, f"  Warning: source chunk persistence failed: {exc}")

        await self._auto_load_source_to_desk(doc)

        # Receipt gate (#85): completion is asserted by receipts, never by the
        # encounter's existence. Doc-complete row -> skip; receipted sections
        # drop out (resume); the enc: sentinel hands back the encounter.
        doc_ref = doc.content_hash
        receipts = await self.store.get_receipts(
            doc_ref, [doc_ref, f"enc:{doc_ref}"] + section_hashes
        )
        if doc_ref in receipts:
            if self.config.verbose:
                _emit(self.config, f"  Already ingested (hash={doc_ref[:8]}...). Skipping.")
            self.last_result["memories_created"] = 0
            return 0
        done_sections = {h for h in section_hashes if h in receipts}
        if done_sections and self.config.verbose:
            _emit(self.config, f"  Resuming: {len(done_sections)}/{len(sections)} sections already ingested.")
        if self.config.verbose:
            _emit(self.config, f"  Mode: {mode.value} | Words: {doc.word_count} | Sections: {len(sections)}")

        # Slow/hybrid mode: delegate to RLM-based ingestion
        if mode in (IngestionMode.SLOW, IngestionMode.HYBRID):
            count = await self._run_rlm_ingest(mode, doc, sections, metrics, llm_calls_start)
            self.last_result["memories_created"] = count
            return count

        # FAST mode: small docs (<=deep_max_words) get per-section appraisal;
        # larger docs get a single doc-level appraisal. All sections processed.
        base_context = await self._build_appraisal_context(doc)
        use_deep = doc.word_count <= self.config.deep_max_words

        # #86: ingestion appraisals no longer overwrite the live affective
        # state (store.set_affective_state used to slam
        # set_current_affective_state -- once per doc here, once per
        # SECTION in the deep branch below, last-write-wins over whatever
        # the agent was actually feeling mid-conversation). Reading still
        # feels like something, but that feeling is recorded on the
        # encounter memory's own emotional_valence -- update_mood()
        # blends it into mood from there over the normal maintenance
        # cycle, the same as any other episodic memory's affect. A
        # doc-level sample is computed unconditionally now (previously
        # only for non-deep docs) so a small document's encounter memory
        # gets its own real appraisal instead of a neutral default.
        sample = self._sample_content(content)
        overall_appraisal = await self.appraiser.appraise(content=sample, context=base_context, mode=mode)
        metrics.appraisal_valence = overall_appraisal.valence
        metrics.appraisal_arousal = overall_appraisal.arousal
        metrics.appraisal_emotion = overall_appraisal.primary_emotion
        metrics.appraisal_intensity = overall_appraisal.intensity

        encounter_id = receipts.get(f"enc:{doc_ref}")
        if encounter_id is None:
            encounter_id = await self._create_encounter_memory(doc, overall_appraisal, mode)
            # The enc: sentinel is a receipt for the encounter only — never a
            # doc-complete claim (#85: a crash from here on RESUMES).
            await self.store.record_receipt(
                doc_ref, f"enc:{doc_ref}",
                memory_id=encounter_id, source_path=doc.path,
            )

        created_ids: list[str] = []
        total_extractions = 0
        dedup_count = 0

        section_hash_by_id = dict(zip((id(s) for s in sections), section_hashes))
        active_sections = [
            s for s in sections
            if not self._skip_section(s.title)
            and section_hash_by_id[id(s)] not in done_sections
        ]
        if _should_cancel(self.config):
            raise RuntimeError("Ingestion cancelled")
        max_items = self.config.max_facts_per_section

        # Bounded LLM fan-out (#90): the rate-limit stampede bound — a large
        # document no longer launches every section's calls at once.
        llm_slots = asyncio.Semaphore(max(1, self.config.max_parallel_llm))

        if use_deep:
            # Small docs: each section gets its own appraisal + extraction,
            # gathered on the caller's loop (#88 — the asyncio.run island died).
            async def _appraise_and_extract(s: Section) -> tuple[Appraisal, list[Extraction]]:
                async with llm_slots:
                    sample = self._sample_content(s.content)
                    apr = await self.appraiser.appraise(content=sample, context=base_context, mode=mode)
                    exts = await self.extractor.extract(
                        section=s, doc=doc, appraisal=apr, mode=mode, max_items=max_items,
                    )
                    return apr, exts

            deep_results = await asyncio.gather(
                *[_appraise_and_extract(s) for s in active_sections]
            )
            for section, (appraisal, extractions) in zip(active_sections, deep_results):
                # #86: per-section appraisals still shape extraction/decay
                # (below) and metrics, but no longer overwrite live
                # affective state -- that was the "30 times in quick
                # succession" thrash this issue is about.
                metrics.appraisal_valence = appraisal.valence
                metrics.appraisal_arousal = appraisal.arousal
                metrics.appraisal_emotion = appraisal.primary_emotion
                metrics.appraisal_intensity = appraisal.intensity
                s_hash = section_hash_by_id[id(section)]
                if not extractions:
                    await self.store.record_receipt(doc_ref, s_hash, source_path=doc.path)
                    continue
                total_extractions += len(extractions)
                new_memories = await self._create_semantic_memories(
                    doc, encounter_id, appraisal, extractions,
                    section_hash=s_hash, chunk_index=section.index)
                dedup_count += len(extractions) - len(new_memories)
                created_ids.extend(new_memories)
        else:
            # Larger docs: shared appraisal, parallel extraction only
            appraisal = overall_appraisal if overall_appraisal is not None else Appraisal()

            async def _bounded_extract(s: Section) -> list[Extraction]:
                async with llm_slots:
                    return await self.extractor.extract(
                        section=s, doc=doc, appraisal=appraisal, mode=mode, max_items=max_items,
                    )

            section_extractions = await asyncio.gather(
                *[_bounded_extract(s) for s in active_sections]
            )
            for section, extractions in zip(active_sections, section_extractions):
                s_hash = section_hash_by_id[id(section)]
                if not extractions:
                    await self.store.record_receipt(doc_ref, s_hash, source_path=doc.path)
                    continue
                total_extractions += len(extractions)
                new_memories = await self._create_semantic_memories(
                    doc, encounter_id, appraisal, extractions,
                    section_hash=s_hash, chunk_index=section.index)
                dedup_count += len(extractions) - len(new_memories)
                created_ids.extend(new_memories)

        if self.config.verbose:
            _emit(self.config, f"  Created {len(created_ids)} semantic memories")

        self.stats["files_processed"] += 1
        self.stats["memories_created"] += len(created_ids) + (1 if encounter_id else 0)

        metrics.extraction_count = total_extractions
        metrics.dedup_count = dedup_count
        metrics.memory_count = len(created_ids) + (1 if encounter_id else 0)
        metrics.llm_calls = self.llm.call_count - llm_calls_start
        metrics.duration_seconds = time.time() - metrics.start_time
        await self.store.store_metrics(metrics)

        # Doc-complete: the final receipt — everything before it is resumable.
        await self.store.record_receipt(
            doc_ref, doc_ref,
            memories_created=len(created_ids), source_path=doc.path,
        )

        self.last_result["memories_created"] = len(created_ids)
        return len(created_ids)

    async def _auto_load_source_to_desk(self, doc: DocumentInfo) -> None:
        """Put newly preserved single-source intake on the RecMem desk.

        Bulk walkers disable config.auto_load_to_desk for the duration of their
        run; connector backfills also skip so large historical imports do not
        flood mid-term working memory.
        """
        source_payload = {
            "source_document_id": doc.document_id,
            "source_document_ids": [doc.document_id] if doc.document_id else [],
            "source_title": doc.title,
            "source_path": doc.path,
            "source_type": doc.source_type,
            "content_hash": doc.content_hash,
        }
        self.last_result.update(source_payload)

        if not doc.document_id:
            self.last_result["desk_load"] = {
                "skipped": True,
                "reason": "missing_source_document_id",
            }
            return

        acquisition = (self.config.acquisition or "user").strip().lower()
        if acquisition == "connector":
            self.last_result["desk_load"] = {
                "skipped": True,
                "reason": "connector_backfill",
            }
            return

        if not self.config.auto_load_to_desk:
            self.last_result["desk_load"] = {
                "skipped": True,
                "reason": "bulk_ingest",
            }
            return

        reason = self.config.acquired_reason or f"ingested source: {doc.title}"
        try:
            self.last_result["desk_load"] = await self.store.load_source_documents_to_recmem(
                [doc.document_id],
                reason=reason,
            )
        except Exception as exc:
            self.last_result["desk_load"] = {"error": str(exc)}
            _emit(self.config, f"  Warning: source preserved but desk load failed: {exc}")

    def _sample_content(self, content: str, limit: int = 2000) -> str:
        if len(content) <= limit:
            return content
        head = content[:limit]
        tail = content[-limit:]
        return f"{head}\n\n...\n\n{tail}"

    async def _build_appraisal_context(self, doc: DocumentInfo) -> dict[str, Any]:
        ctx = {
            "document": {
                "title": doc.title,
                "source_type": doc.source_type,
                "word_count": doc.word_count,
            }
        }
        try:
            ctx.update(await self.store.fetch_appraisal_context())
        except Exception:
            pass
        return ctx

    def _source_payload(
        self,
        doc: DocumentInfo,
        *,
        section_hash: str | None = None,
        chunk_index: int | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        payload = {
            "kind": doc.source_type,
            "ref": doc.content_hash,
            "label": doc.title,
            "observed_at": now,
            "content_hash": doc.content_hash,
            "path": doc.path,
        }
        if section_hash is not None:
            # The persist functions record the section receipt atomically
            # with persistence when this key rides the source (#85/#90).
            payload["section_hash"] = section_hash
        if doc.document_id:
            payload["source_document_id"] = doc.document_id
            payload["document_id"] = doc.document_id
        if chunk_index is not None:
            chunk_id = doc.chunk_ids.get(chunk_index)
            if chunk_id:
                payload["chunk_id"] = chunk_id
                payload["chunk_index"] = chunk_index
        if self.config.base_trust is not None:
            payload["trust"] = float(self.config.base_trust)
        if self.config.sensitivity:
            payload["sensitivity"] = str(self.config.sensitivity)
        if self.config.acquisition:
            payload["acquisition"] = str(self.config.acquisition)
        if self.config.acquired_reason:
            payload["acquired_reason"] = str(self.config.acquired_reason)
        return payload

    async def _create_archive_encounter(self, doc: DocumentInfo) -> str | None:
        source = self._source_payload(doc)
        text = f"I have access to '{doc.title}' but haven't engaged with it yet."
        context = {
            "activity": "archived",
            "source_type": doc.source_type,
            "source_ref": doc.content_hash,
            "word_count": doc.word_count,
            "mode": "archived",
            "awaiting_processing": True,
        }
        importance = max(self.config.min_importance_floor or 0.0, 0.2)
        encounter_id = await self.store.create_encounter_memory(
            text=text,
            source=source,
            emotional_valence=0.0,
            context=context,
            importance=importance,
        )
        await self._apply_decay(encounter_id, intensity=0.0)
        return encounter_id

    async def _create_encounter_memory(self, doc: DocumentInfo, appraisal: Appraisal | None, mode: IngestionMode) -> str | None:
        source = self._source_payload(doc)
        appraisal = appraisal or Appraisal()
        summary = appraisal.summary or ""
        if not summary:
            summary = f"It felt {appraisal.primary_emotion} with intensity {appraisal.intensity:.2f}."
        text = f"I read '{doc.title}'. {summary}"
        context = {
            "activity": "reading",
            "source_type": doc.source_type,
            "source_ref": doc.content_hash,
            "word_count": doc.word_count,
            "mode": mode.value,
            "appraisal": appraisal.__dict__,
        }
        importance = max(self.config.min_importance_floor or 0.0, 0.4 + appraisal.intensity * 0.4)
        encounter_id = await self.store.create_encounter_memory(
            text=text,
            source=source,
            emotional_valence=appraisal.valence,
            context=context,
            importance=importance,
        )
        await self._apply_decay(encounter_id, intensity=appraisal.intensity)
        return encounter_id

    async def _apply_decay(self, memory_id: str, intensity: float) -> None:
        # Decay bands are DB-owned (db/66 decay_rate_for_intensity).
        await self.store.apply_ingest_decay(memory_id, intensity, self.config.permanent)

    async def _create_semantic_memories(
        self,
        doc: DocumentInfo,
        encounter_id: str | None,
        appraisal: Appraisal,
        extractions: list[Extraction],
        *,
        section_hash: str | None = None,
        chunk_index: int | None = None,
    ) -> list[str]:
        """Thin wrapper: the whole post-LLM persistence pass runs atomically
        in the DB (db/66 ingest_persist_extractions) — routing, corroboration
        via the audited belief-revision policy, creation, concept links,
        worldview-hint edges, provenance edges, decay, and the section
        receipt (#85)."""
        source = self._source_payload(doc, section_hash=section_hash, chunk_index=chunk_index)
        result = await self.store.persist_extractions(
            extractions,
            source,
            encounter_id=encounter_id,
            intensity=appraisal.intensity,
            min_confidence=self.config.min_confidence_threshold,
            min_importance_floor=self.config.min_importance_floor,
            base_trust=self.config.base_trust,
            permanent=self.config.permanent,
        )
        return [str(mid) for mid in (result.get("created") or [])]

    def _skip_section(self, title: str) -> bool:
        lowered = title.strip().lower()
        return any(skip in lowered for skip in self.config.skip_sections)

    async def _run_rlm_ingest(
        self,
        mode: IngestionMode,
        doc: DocumentInfo,
        sections: list[Section],
        metrics: IngestionMetrics,
        llm_calls_start: int,
    ) -> int:
        """Run slow or hybrid ingestion via the async RLM loop, on the
        caller's loop (#88 — the thread-plus-third-loop bridge died)."""
        from services.slow_ingest_rlm import run_hybrid_ingest, run_slow_ingest

        if self.config.dsn:
            dsn = self.config.dsn
        else:
            dsn = (
                f"postgresql://{self.config.db_user}:{self.config.db_password}"
                f"@{self.config.db_host}:{self.config.db_port}/{self.config.db_name}"
            )

        runner = run_slow_ingest if mode == IngestionMode.SLOW else run_hybrid_ingest
        result = await runner(
            pipeline=self,
            doc=doc,
            sections=sections,
            llm_config=self.llm._cfg,
            dsn=dsn,
        )

        count = result.get("memories_created", 0)
        self.stats["files_processed"] += 1
        self.stats["memories_created"] += count

        metrics.memory_count = count
        metrics.mode = mode.value
        metrics.llm_calls = self.llm.call_count - llm_calls_start
        metrics.duration_seconds = time.time() - metrics.start_time
        await self.store.store_metrics(metrics)

        if self.config.verbose:
            _emit(self.config, f"  RLM {mode.value} ingest: {count} memories created")
            if mode == IngestionMode.HYBRID:
                _emit(
                    self.config,
                    f"  Slow chunks: {result.get('slow_chunks', 0)} | "
                    f"Fast chunks: {result.get('fast_chunks', 0)}",
                )

        return count

    async def check_and_process_archived(self, query: str, threshold: float = 0.75) -> list[str]:
        """
        Check if any archived content matches the query and process it.

        This implements retrieval-triggered processing: when a query surfaces
        archived content that hasn't been fully processed, we upgrade it now.

        Returns list of content hashes that were processed.
        """
        if self.store.client is None:
            await self.store.connect()

        # Find archived content matching the query
        rows = await self.store._fetchval(
            """
            SELECT jsonb_agg(jsonb_build_object(
                'memory_id', memory_id,
                'content_hash', content_hash,
                'title', title,
                'similarity', similarity,
                'source_path', source_path
            ))
            FROM check_archived_for_query($1, $2, 5)
            """,
            query,
            threshold,
        )

        if not rows:
            return []

        archived = json.loads(rows) if isinstance(rows, str) else rows
        if not archived:
            return []

        processed_hashes: list[str] = []

        for item in archived:
            if not item:
                continue

            content_hash = item.get("content_hash")
            source_path = item.get("source_path")
            title = item.get("title")
            memory_id = item.get("memory_id")

            if not content_hash:
                continue

            _emit(self.config, f"Processing archived content triggered by query: {title}")

            # Attempt to re-read the source file if it exists
            if source_path and source_path != "stdin" and not source_path.startswith("http"):
                path = Path(source_path)
                if path.exists():
                    # Re-ingest the file with the current mode (not archive)
                    original_mode = self.config.mode
                    self.config.mode = IngestionMode.FAST
                    try:
                        # Mark as processed first to avoid duplicate detection
                        await self.store._fetchval(
                            "SELECT mark_archived_as_processed($1::uuid)",
                            memory_id,
                        )
                        await self.ingest_file(path)
                        processed_hashes.append(content_hash)
                    finally:
                        self.config.mode = original_mode
                    continue

            # If source file not available, just mark as processed
            await self.store._fetchval(
                "SELECT mark_archived_as_processed($1::uuid)",
                memory_id,
            )
            processed_hashes.append(content_hash)

        return processed_hashes

    def print_stats(self) -> None:
        _emit(self.config, "\n" + "=" * 50)
        _emit(self.config, "INGESTION COMPLETE")
        _emit(self.config, "=" * 50)
        _emit(self.config, f"Files processed:   {self.stats['files_processed']}")
        _emit(self.config, f"Memories created:  {self.stats['memories_created']}")
        _emit(self.config, f"Errors:            {self.stats['errors']}")
        _emit(self.config, "=" * 50)

    async def close(self) -> None:
        await self.store.close()


# =========================================================================
# ARCHIVED CONTENT PROCESSOR
# =========================================================================


class ArchivedContentProcessor:
    """
    Processor for upgrading archived content to full memories.

    This can be used:
    1. During recall - when a query surfaces relevant archived content
    2. By maintenance workers - batch processing of pending archives
    3. By CLI - manual processing of specific content
    """

    def __init__(self, config: Config):
        self.config = config
        self.pipeline = IngestionPipeline(config)

    async def process_for_query(self, query: str, threshold: float = 0.75) -> list[str]:
        """
        Check if archived content matches a query and process it.

        Returns list of content hashes that were processed.
        """
        return await self.pipeline.check_and_process_archived(query, threshold)

    async def process_by_hash(self, content_hash: str) -> bool:
        """Process a specific archived item by content hash."""
        archived = await self.pipeline.store.check_archived_for_query(
            content_hash, threshold=0.0, limit=1
        )

        if not archived:
            # Try direct lookup
            row = await self.pipeline.store._fetchval(
                """
                SELECT jsonb_build_object(
                    'memory_id', id,
                    'content_hash', source_attribution->>'content_hash',
                    'title', source_attribution->>'label',
                    'source_path', source_attribution->>'path'
                )
                FROM memories
                WHERE type = 'episodic'
                  AND source_attribution->>'content_hash' = $1
                  AND metadata->>'awaiting_processing' = 'true'
                LIMIT 1
                """,
                content_hash,
            )
            if not row:
                return False
            archived = [json.loads(row) if isinstance(row, str) else row]

        for item in archived:
            if not item:
                continue

            source_path = item.get("source_path")
            memory_id = item.get("memory_id")

            if source_path and source_path != "stdin" and not source_path.startswith("http"):
                path = Path(source_path)
                if path.exists():
                    await self.pipeline.store.mark_archived_processed(memory_id)
                    original_mode = self.config.mode
                    self.config.mode = IngestionMode.FAST
                    try:
                        await self.pipeline.ingest_file(path)
                    finally:
                        self.config.mode = original_mode
                    return True

            # Mark as processed even if file not found
            return await self.pipeline.store.mark_archived_processed(memory_id)

        return False

    async def process_batch(self, limit: int = 10) -> int:
        """Process a batch of archived items."""
        rows = await self.pipeline.store._fetchval(
            """
            SELECT ARRAY_AGG(source_attribution->>'content_hash')
            FROM memories
            WHERE type = 'episodic'
              AND metadata->>'awaiting_processing' = 'true'
            ORDER BY importance DESC, created_at ASC
            LIMIT $1
            """,
            limit,
        )

        if not rows:
            return 0

        hashes = list(rows) if rows else []
        count = 0
        for h in hashes:
            if h and await self.process_by_hash(h):
                count += 1

        return count

    async def close(self) -> None:
        await self.pipeline.close()
