-- Attachments are readable in the turn they are attached.
--
-- The composer preserves the file and asks the API to read it immediately,
-- so "here is a PDF, what do you think?" is answerable in the same breath.
-- These caps bound that synchronous read; the durable ingestion job still
-- reads the same bytes afterwards with no budget at all.
SET search_path = public, ag_catalog, "$user";

INSERT INTO config_defaults (key, value, description) VALUES
    ('ingest.attachment_text_chars', '60000'::jsonb,
     'Maximum characters of an attached file''s text carried into the turn it is attached to'),
    ('ingest.attachment_read_timeout_s', '25'::jsonb,
     'Seconds the composer waits for an attached file''s text before leaving it to background ingestion'),
    ('ingest.attachment_read_max_bytes', '26214400'::jsonb,
     'Largest attachment read synchronously at attach time; bigger files go straight to background ingestion')
ON CONFLICT (key) DO NOTHING;
