-- HMX canonical digests: independent PL/pgSQL implementation of the open
-- standard defined in plans/hmx.md ("Canonical Hashing" / "Canonical JSON
-- Serialization v1"). core/digest.py is the Python reference implementation;
-- this file exists to prove the byte contract is language-neutral and to give
-- the database its own digest capability. Both implementations must reproduce
-- tests/fixtures/digest/ byte-for-byte.

-- Each baseline file runs as its own psql session at initdb, so it must pin
-- its own search_path: under an ag_catalog-first cluster default, unqualified
-- CREATEs would land Hexis objects in ag_catalog (the #77 fossil bug).
SET search_path = public, ag_catalog, "$user";

SET check_function_bodies = off;

-- ---------------------------------------------------------------------------
-- content_hash_v1
-- ---------------------------------------------------------------------------

-- normalize_v1 whitespace is the explicit code point set from the spec
-- (Unicode White_Space plus the U+001C..U+001F separators) -- never the
-- regex engine's own notion of \s, which differs between languages.
CREATE OR REPLACE FUNCTION hmx_normalize_v1(p_content TEXT)
RETURNS TEXT AS $$
DECLARE
    ws CONSTANT TEXT := '[\u0009-\u000d\u001c-\u001f\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]';
    trimmed TEXT;
BEGIN
    trimmed := regexp_replace(coalesce(p_content, ''), '^' || ws || '+|' || ws || '+$', '', 'g');
    RETURN lower(regexp_replace(trimmed, ws || '+', ' ', 'g'));
END;
$$ LANGUAGE plpgsql IMMUTABLE;

CREATE OR REPLACE FUNCTION hmx_content_hash_v1(p_content TEXT)
RETURNS TEXT AS $$
    SELECT encode(digest(convert_to(hmx_normalize_v1(p_content), 'UTF8'), 'sha256'), 'hex');
$$ LANGUAGE sql IMMUTABLE;

-- ---------------------------------------------------------------------------
-- Canonical JSON Serialization v1: numbers
-- ---------------------------------------------------------------------------

-- Exact decimal value of a binary64, via its IEEE-754 bits. numeric holds it
-- exactly (worst case ~751 significant digits for subnormals).
CREATE OR REPLACE FUNCTION hmx_float8_exact_numeric(p_value FLOAT8)
RETURNS NUMERIC AS $$
DECLARE
    b BYTEA := float8send(p_value);
    sign_bit INT := get_byte(b, 0) >> 7;
    exp_biased INT := ((get_byte(b, 0) & 127) << 4) | (get_byte(b, 1) >> 4);
    mant BIGINT := (get_byte(b, 1) & 15)::BIGINT;
    m NUMERIC;
    e2 INT;
    pow NUMERIC := 1;
    digits TEXT;
    point_at INT;
    result NUMERIC;
    i INT;
BEGIN
    FOR i IN 2..7 LOOP
        mant := mant * 256 + get_byte(b, i);
    END LOOP;
    IF exp_biased = 2047 THEN
        RAISE EXCEPTION 'non-finite numbers are not valid HMX canonical JSON';
    END IF;
    IF exp_biased = 0 THEN
        m := mant::NUMERIC;              -- subnormal (or zero)
        e2 := -1074;
    ELSE
        m := mant::NUMERIC + 4503599627370496::NUMERIC;  -- 2^52 implicit bit
        e2 := exp_biased - 1075;
    END IF;
    IF m = 0 THEN
        RETURN 0;
    END IF;

    IF e2 >= 0 THEN
        FOR i IN 1..e2 LOOP
            pow := pow * 2;
        END LOOP;
        result := m * pow;
    ELSE
        -- m * 2^e2 = (m * 5^k) / 10^k with k = -e2; the division is an exact
        -- decimal-point placement done on the digit string.
        FOR i IN 1..(-e2) LOOP
            pow := pow * 5;
        END LOOP;
        digits := (m * pow)::TEXT;
        point_at := length(digits) + e2;   -- e2 < 0
        IF point_at <= 0 THEN
            digits := '0.' || repeat('0', -point_at) || digits;
        ELSE
            digits := substr(digits, 1, point_at) || '.' || substr(digits, point_at + 1);
        END IF;
        result := digits::NUMERIC;
    END IF;
    RETURN CASE WHEN sign_bit = 1 THEN -result ELSE result END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Round a binary64's exact value to the nearest multiple of 10^-6, ties to
-- even, exactly (spec rule 6a). Returns the rounded value as numeric.
CREATE OR REPLACE FUNCTION hmx_round6_ties_even(p_value FLOAT8)
RETURNS NUMERIC AS $$
DECLARE
    exact NUMERIC := hmx_float8_exact_numeric(p_value);
    neg BOOLEAN := exact < 0;
    scaled NUMERIC := abs(exact) * 1000000;
    fl NUMERIC := trunc(scaled);
    frac NUMERIC := scaled - fl;
    result NUMERIC;
BEGIN
    IF frac > 0.5 OR (frac = 0.5 AND fl % 2 = 1) THEN
        fl := fl + 1;
    END IF;
    result := fl * 0.000001;
    RETURN CASE WHEN neg THEN -result ELSE result END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Serialize a non-integer number per spec rule 6. The shortest round-trip
-- representation is generated directly from the spec definition: for
-- k = 1..17 significant digits, round the value's exact decimal to k digits
-- (ties to even) and accept the first candidate that converts back to the
-- identical binary64 (input conversion is correctly rounded, ties to even).
-- This never depends on PostgreSQL's float8 output formatting.
CREATE OR REPLACE FUNCTION hmx_canonical_float_v1(p_value FLOAT8)
RETURNS TEXT AS $$
DECLARE
    r FLOAT8;
    r_bits BYTEA;
    exact NUMERIC;
    neg BOOLEAN;
    txt TEXT;
    ip TEXT;
    fp TEXT;
    all_digits TEXT;
    sig TEXT;
    lead INT;
    e10 INT;
    k INT;
    kept TEXT;
    rest TEXT;
    rest_sig TEXT;
    round_up BOOLEAN;
    bumped TEXT;
    digits TEXT;
    e INT;
    n INT;
    cand_text TEXT;
    exp_text TEXT;
BEGIN
    r := hmx_round6_ties_even(p_value)::FLOAT8;
    IF r = 0 THEN
        RETURN '0.0';
    END IF;
    r_bits := float8send(r);
    exact := abs(hmx_float8_exact_numeric(r));
    neg := r < 0;

    -- Significant digits of the exact decimal (numeric text never uses
    -- exponent notation) and the power-of-ten position of the first digit.
    txt := exact::TEXT;
    ip := split_part(txt, '.', 1);
    fp := split_part(txt, '.', 2);
    all_digits := ip || fp;
    lead := length(all_digits) - length(ltrim(all_digits, '0'));
    sig := rtrim(ltrim(all_digits, '0'), '0');
    e10 := length(ip) - lead - 1;  -- first significant digit is d0 x 10^e10

    digits := NULL;
    FOR k IN 1..17 LOOP
        IF k >= length(sig) THEN
            kept := sig;
            e := e10;
        ELSE
            kept := substr(sig, 1, k);
            rest := substr(sig, k + 1);
            rest_sig := rtrim(rest, '0');
            IF substr(rest, 1, 1) > '5' THEN
                round_up := true;
            ELSIF substr(rest, 1, 1) < '5' THEN
                round_up := false;
            ELSIF length(rest_sig) > 1 THEN  -- 5 followed by a nonzero digit
                round_up := true;
            ELSE                              -- exact tie: round to even
                round_up := (substr(kept, k, 1)::INT % 2) = 1;
            END IF;
            e := e10;
            IF round_up THEN
                bumped := ((kept::NUMERIC) + 1)::TEXT;
                IF length(bumped) > k THEN    -- 99..9 -> 100..0 carries a power
                    e := e + 1;
                    bumped := substr(bumped, 1, k);
                END IF;
                kept := bumped;
            END IF;
            kept := rtrim(kept, '0');
            IF kept = '' THEN  -- defensive; nonzero value keeps a digit
                kept := '1';
            END IF;
        END IF;
        cand_text := substr(kept, 1, 1)
            || CASE WHEN length(kept) > 1 THEN '.' || substr(kept, 2) ELSE '' END
            || 'e' || e::TEXT;
        BEGIN
            IF float8send((CASE WHEN neg THEN '-' ELSE '' END || cand_text)::FLOAT8) = r_bits THEN
                digits := kept;
                EXIT;
            END IF;
        EXCEPTION WHEN numeric_value_out_of_range THEN
            NULL;  -- rounded-up candidate overflowed the range; try more digits
        END;
    END LOOP;
    IF digits IS NULL THEN  -- defensive; 17 digits always round-trip
        RAISE EXCEPTION 'no round-trip representation found for %', r;
    END IF;
    n := length(digits) - 1;

    IF e >= -4 AND e < 16 THEN
        IF e >= n THEN
            txt := digits || repeat('0', e - n) || '.0';
        ELSIF e >= 0 THEN
            txt := substr(digits, 1, e + 1) || '.' || substr(digits, e + 2);
        ELSE
            txt := '0.' || repeat('0', -e - 1) || digits;
        END IF;
    ELSE
        exp_text := abs(e)::TEXT;
        IF length(exp_text) < 2 THEN
            exp_text := '0' || exp_text;
        END IF;
        txt := substr(digits, 1, 1)
            || CASE WHEN n >= 1 THEN '.' || substr(digits, 2) ELSE '' END
            || 'e' || CASE WHEN e < 0 THEN '-' ELSE '+' END
            || exp_text;
    END IF;
    RETURN CASE WHEN neg THEN '-' ELSE '' END || txt;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ---------------------------------------------------------------------------
-- Canonical JSON Serialization v1: strings and values
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION hmx_canonical_string_v1(p_text TEXT)
RETURNS TEXT AS $$
DECLARE
    out TEXT := '"';
    c TEXT;
    cp INT;
    cp2 INT;
    i INT;
BEGIN
    FOR i IN 1..coalesce(length(p_text), 0) LOOP
        c := substr(p_text, i, 1);
        cp := ascii(c);
        IF c = '"' THEN
            out := out || '\"';
        ELSIF c = '\' THEN
            out := out || '\\';
        ELSIF cp = 8 THEN
            out := out || '\b';
        ELSIF cp = 9 THEN
            out := out || '\t';
        ELSIF cp = 10 THEN
            out := out || '\n';
        ELSIF cp = 12 THEN
            out := out || '\f';
        ELSIF cp = 13 THEN
            out := out || '\r';
        ELSIF cp < 32 THEN
            out := out || '\u' || lpad(to_hex(cp), 4, '0');
        ELSIF cp > 126 THEN
            IF cp > 65535 THEN
                cp2 := cp - 65536;
                out := out || '\u' || lpad(to_hex(55296 + (cp2 >> 10)), 4, '0')
                           || '\u' || lpad(to_hex(56320 + (cp2 & 1023)), 4, '0');
            ELSE
                out := out || '\u' || lpad(to_hex(cp), 4, '0');
            END IF;
        ELSE
            out := out || c;
        END IF;
    END LOOP;
    RETURN out || '"';
END;
$$ LANGUAGE plpgsql IMMUTABLE;

CREATE OR REPLACE FUNCTION hmx_canonical_json_v1(p_doc JSONB)
RETURNS TEXT AS $$
DECLARE
    kind TEXT := jsonb_typeof(p_doc);
    out TEXT;
    key TEXT;
    item JSONB;
    first BOOLEAN := true;
    num_text TEXT;
BEGIN
    IF kind = 'object' THEN
        out := '{';
        FOR key IN SELECT k FROM jsonb_object_keys(p_doc) AS t(k) ORDER BY k COLLATE "C" LOOP
            IF NOT first THEN
                out := out || ',';
            END IF;
            first := false;
            out := out || hmx_canonical_string_v1(key) || ':' || hmx_canonical_json_v1(p_doc -> key);
        END LOOP;
        RETURN out || '}';
    ELSIF kind = 'array' THEN
        out := '[';
        FOR item IN SELECT e FROM jsonb_array_elements(p_doc) AS t(e) LOOP
            IF NOT first THEN
                out := out || ',';
            END IF;
            first := false;
            out := out || hmx_canonical_json_v1(item);
        END LOOP;
        RETURN out || ']';
    ELSIF kind = 'string' THEN
        RETURN hmx_canonical_string_v1(p_doc #>> '{}');
    ELSIF kind = 'number' THEN
        num_text := p_doc #>> '{}';
        IF position('.' in num_text) = 0 AND position('e' in lower(num_text)) = 0 THEN
            RETURN CASE WHEN num_text = '-0' THEN '0' ELSE num_text END;
        END IF;
        RETURN hmx_canonical_float_v1((num_text::NUMERIC)::FLOAT8);
    ELSIF kind = 'boolean' OR kind = 'null' THEN
        RETURN p_doc::TEXT;
    END IF;
    RAISE EXCEPTION 'unsupported jsonb type %', kind;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ---------------------------------------------------------------------------
-- protected_section_digest_v1
-- ---------------------------------------------------------------------------

-- Field exclusion (spec: "Digest field inclusion and exclusion").
CREATE OR REPLACE FUNCTION hmx_digest_strip_excluded_v1(p_value JSONB)
RETURNS JSONB AS $$
DECLARE
    kind TEXT := jsonb_typeof(p_value);
    out JSONB;
    key TEXT;
    item JSONB;
    child JSONB;
BEGIN
    IF kind = 'object' THEN
        out := '{}'::jsonb;
        FOR key, item IN SELECT k, v FROM jsonb_each(p_value) AS t(k, v) LOOP
            IF key IN ('ref', 'export_id', 'import_chain', 'modification_chain',
                       'access_count', 'last_accessed', 'created_at', 'updated_at',
                       'hmx_id', 'blocked_by', 'parent_goal_id', 'provenance')
               OR key LIKE '\_transient\_%'
               OR key LIKE '%\_ref'
               OR key LIKE '%\_refs' THEN
                CONTINUE;
            END IF;
            IF key = 'metadata' AND jsonb_typeof(item) = 'object' THEN
                child := '{}'::jsonb;
                child := (
                    SELECT coalesce(jsonb_object_agg(k, v), '{}'::jsonb)
                    FROM jsonb_each(item) AS m(k, v)
                    WHERE k NOT IN ('hmx', 'unrecognized_hmx_fields')
                      AND k NOT LIKE 'embedding\_%'
                );
                item := child;
            END IF;
            out := out || jsonb_build_object(key, hmx_digest_strip_excluded_v1(item));
        END LOOP;
        RETURN out;
    ELSIF kind = 'array' THEN
        RETURN coalesce(
            (SELECT jsonb_agg(hmx_digest_strip_excluded_v1(e)) FROM jsonb_array_elements(p_value) AS t(e)),
            '[]'::jsonb
        );
    END IF;
    RETURN p_value;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

CREATE OR REPLACE FUNCTION hmx_digest_record_hash_v1(p_record JSONB)
RETURNS TEXT AS $$
    SELECT encode(digest(convert_to(hmx_canonical_json_v1(p_record), 'UTF8'), 'sha256'), 'hex');
$$ LANGUAGE sql IMMUTABLE;

-- Prune one record; identity records also drop derived life_chapter_current
-- facets and sort remaining facets by concept.
CREATE OR REPLACE FUNCTION hmx_digest_prepare_record_v1(p_section TEXT, p_record JSONB)
RETURNS JSONB AS $$
DECLARE
    pruned JSONB := hmx_digest_strip_excluded_v1(p_record);
    facets JSONB;
BEGIN
    IF p_section = 'identity' AND jsonb_typeof(pruned) = 'object'
       AND jsonb_typeof(pruned -> 'facets') = 'array' THEN
        facets := coalesce(
            (
                SELECT jsonb_agg(f ORDER BY coalesce(f ->> 'concept', '') COLLATE "C")
                FROM jsonb_array_elements(pruned -> 'facets') AS t(f)
                WHERE NOT (
                    jsonb_typeof(f) = 'object'
                    AND coalesce(f ->> 'kind', f ->> 'type', '') = 'life_chapter_current'
                )
            ),
            '[]'::jsonb
        );
        pruned := jsonb_set(pruned, '{facets}', facets);
    END IF;
    RETURN pruned;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Sort key: semantic key -> provenance.origin_id (from the ORIGINAL record) ->
-- canonical record hash; the record hash is always the final tiebreak.
CREATE OR REPLACE FUNCTION hmx_digest_sort_key_v1(
    p_section TEXT,
    p_original JSONB,
    p_pruned JSONB,
    OUT sort_key TEXT,
    OUT record_hash TEXT
) AS $$
DECLARE
    semantic TEXT;
BEGIN
    record_hash := hmx_digest_record_hash_v1(p_pruned);
    IF jsonb_typeof(p_pruned) = 'object' THEN
        IF p_section = 'identity' AND coalesce(p_pruned ->> 'key', '') <> '' THEN
            semantic := p_pruned ->> 'key';
        ELSIF p_section = 'worldview' AND coalesce(p_pruned ->> 'content', '') <> '' THEN
            semantic := hmx_content_hash_v1(p_pruned ->> 'content');
        ELSIF p_section = 'goals' AND coalesce(p_pruned ->> 'title', '') <> '' THEN
            semantic := hmx_content_hash_v1((p_pruned ->> 'title') || coalesce(p_pruned ->> 'description', ''));
        ELSIF p_section = 'drives' AND coalesce(p_pruned ->> 'name', '') <> '' THEN
            semantic := p_pruned ->> 'name';
        ELSIF p_section = 'emotional_triggers' AND coalesce(p_pruned ->> 'trigger_pattern', '') <> '' THEN
            semantic := hmx_content_hash_v1(p_pruned ->> 'trigger_pattern');
        END IF;
    END IF;
    IF semantic IS NOT NULL THEN
        sort_key := semantic;
        RETURN;
    END IF;
    IF jsonb_typeof(p_original) = 'object'
       AND coalesce(p_original #>> '{provenance,origin_id}', '') <> '' THEN
        sort_key := p_original #>> '{provenance,origin_id}';
        RETURN;
    END IF;
    sort_key := record_hash;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

CREATE OR REPLACE FUNCTION hmx_protected_section_canonical_v1(p_section TEXT, p_data JSONB)
RETURNS TEXT AS $$
DECLARE
    records JSONB;
    canonical JSONB;
    subsection TEXT;
    entries JSONB;
    prepared JSONB;
    out TEXT;
    first BOOLEAN := true;
BEGIN
    IF p_section = 'narrative' THEN
        IF jsonb_typeof(p_data) <> 'object' THEN
            RAISE EXCEPTION 'narrative section_data must be an object of subsection arrays';
        END IF;
        out := '{';
        FOR subsection IN SELECT k FROM jsonb_object_keys(p_data) AS t(k) ORDER BY k COLLATE "C" LOOP
            entries := p_data -> subsection;
            IF jsonb_typeof(entries) <> 'array' THEN
                entries := jsonb_build_array(entries);
            END IF;
            IF subsection = 'life_chapters' THEN
                prepared := coalesce(
                    (
                        SELECT jsonb_agg(hmx_digest_prepare_record_v1('narrative', e) ORDER BY ord)
                        FROM jsonb_array_elements(entries) WITH ORDINALITY AS t(e, ord)
                    ),
                    '[]'::jsonb
                );
            ELSE
                prepared := coalesce(
                    (
                        SELECT jsonb_agg(p ORDER BY hmx_digest_record_hash_v1(p) COLLATE "C")
                        FROM (
                            SELECT hmx_digest_prepare_record_v1('narrative', e) AS p
                            FROM jsonb_array_elements(entries) AS t(e)
                        ) sub
                    ),
                    '[]'::jsonb
                );
            END IF;
            IF NOT first THEN
                out := out || ',';
            END IF;
            first := false;
            out := out || hmx_canonical_string_v1(subsection) || ':' || hmx_canonical_json_v1(prepared);
        END LOOP;
        RETURN out || '}';
    END IF;

    IF jsonb_typeof(p_data) = 'object' THEN
        records := jsonb_build_array(p_data);
    ELSIF jsonb_typeof(p_data) = 'array' THEN
        records := p_data;
    ELSE
        RAISE EXCEPTION 'section_data for % must be an array or object', p_section;
    END IF;

    canonical := coalesce(
        (
            SELECT jsonb_agg(pruned ORDER BY (keys).sort_key COLLATE "C", (keys).record_hash COLLATE "C")
            FROM (
                SELECT hmx_digest_prepare_record_v1(p_section, e) AS pruned,
                       hmx_digest_sort_key_v1(p_section, e, hmx_digest_prepare_record_v1(p_section, e)) AS keys
                FROM jsonb_array_elements(records) AS t(e)
            ) sub
        ),
        '[]'::jsonb
    );
    RETURN hmx_canonical_json_v1(canonical);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

CREATE OR REPLACE FUNCTION hmx_protected_section_digest_v1(p_section TEXT, p_data JSONB)
RETURNS TEXT AS $$
    SELECT encode(digest(convert_to(hmx_protected_section_canonical_v1(p_section, p_data), 'UTF8'), 'sha256'), 'hex');
$$ LANGUAGE sql IMMUTABLE;

-- ---------------------------------------------------------------------------
-- audit_record_digest_v1
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION hmx_audit_record_digest_v1(p_record JSONB)
RETURNS TEXT AS $$
DECLARE
    pruned JSONB := p_record - 'audit_id' - 'imported_at' - 'local_record_id';
    metadata JSONB;
BEGIN
    IF jsonb_typeof(pruned -> 'metadata') = 'object' THEN
        metadata := (pruned -> 'metadata') - 'unrecognized_hmx_fields';
        IF metadata = '{}'::jsonb THEN
            pruned := pruned - 'metadata';
        ELSE
            pruned := jsonb_set(pruned, '{metadata}', metadata);
        END IF;
    END IF;
    RETURN encode(digest(convert_to(hmx_canonical_json_v1(pruned), 'UTF8'), 'sha256'), 'hex');
END;
$$ LANGUAGE plpgsql IMMUTABLE;

SET check_function_bodies = on;
