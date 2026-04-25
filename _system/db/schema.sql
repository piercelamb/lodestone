-- ============================================================
-- GROUP 1: CORE CONTENT
-- ============================================================

CREATE TABLE IF NOT EXISTS domains (
    name TEXT PRIMARY KEY,
    description TEXT
);

CREATE TABLE IF NOT EXISTS collections (
    domain TEXT NOT NULL REFERENCES domains(name),
    name TEXT NOT NULL,
    description TEXT,
    PRIMARY KEY(domain, name)
);

CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY,
    arxiv_id TEXT UNIQUE NOT NULL,
    paper_name TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    authors TEXT NOT NULL,
    date TEXT NOT NULL,
    abstract TEXT NOT NULL,
    -- domain/content_hash are filled in by later pipeline stages.
    -- domain is set by classify_paper.py; content_hash is NULL only for
    -- FAILED_HTML stubs (no PDF was ever downloaded).
    domain TEXT REFERENCES domains(name),
    collection TEXT,
    code_repo TEXT,
    content_hash TEXT,
    pdf_url TEXT NOT NULL,
    html_source TEXT,
    ingested_at TEXT NOT NULL,
    status TEXT NOT NULL,
    markdown TEXT,
    raw_html TEXT,
    section_count INTEGER DEFAULT 0,
    entity_count INTEGER DEFAULT 0,
    figure_count INTEGER DEFAULT 0,
    needs_review INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_papers_domain ON papers(domain);
CREATE INDEX IF NOT EXISTS idx_papers_collection ON papers(domain, collection);
CREATE INDEX IF NOT EXISTS idx_papers_hash ON papers(content_hash);
CREATE INDEX IF NOT EXISTS idx_papers_review ON papers(needs_review) WHERE needs_review = 1;

CREATE TABLE IF NOT EXISTS figures (
    id INTEGER PRIMARY KEY,
    paper_id INTEGER NOT NULL REFERENCES papers(id),
    figure_number INTEGER NOT NULL,
    display_number TEXT,
    figure_id TEXT,
    caption TEXT NOT NULL,
    section_context TEXT,
    image BLOB NOT NULL,
    mime_type TEXT NOT NULL DEFAULT 'image/png',
    UNIQUE(paper_id, figure_number)
);

CREATE TABLE IF NOT EXISTS page_images (
    paper_id INTEGER NOT NULL REFERENCES papers(id),
    page_number INTEGER NOT NULL,
    image BLOB NOT NULL,
    PRIMARY KEY(paper_id, page_number)
);

CREATE TABLE IF NOT EXISTS paper_references (
    id INTEGER PRIMARY KEY,
    paper_id INTEGER NOT NULL REFERENCES papers(id),
    -- Source-side identity. bibitem_id is the LaTeXML anchor target
    -- ("bib.bib28") for standard `<li class="ltx_bibitem">` extractions;
    -- NULL for hand-typed papers where references are bare `<p>` paragraphs
    -- in a "References" section. ref_number is the [N] from a hand-typed
    -- paper or the 1-based bibitem position for the standard regime, so
    -- query-time `\[(\d+)\]` lookups are uniform across both shapes.
    bibitem_id TEXT,
    ref_number INTEGER NOT NULL,
    raw_text TEXT NOT NULL,
    -- Resolved-target identity. cited_arxiv_id is the bare canonical form
    -- (e.g. "2310.08560" or legacy "cs/0701006"), normalized from the
    -- `arXiv:...` / `arxiv.org/abs/...` token regex. NULL when the
    -- reference contains no arxiv-id (NeurIPS-only paper, missing eprint
    -- field, etc.). cited_paper_id is set when cited_arxiv_id matches a
    -- row in papers; otherwise NULL.
    cited_arxiv_id TEXT,
    cited_paper_id INTEGER REFERENCES papers(id),
    UNIQUE(paper_id, ref_number)
);

CREATE INDEX IF NOT EXISTS idx_paper_refs_paper ON paper_references(paper_id);
CREATE INDEX IF NOT EXISTS idx_paper_refs_cited_arxiv ON paper_references(cited_arxiv_id);
CREATE INDEX IF NOT EXISTS idx_paper_refs_cited_paper ON paper_references(cited_paper_id);

-- ============================================================
-- GROUP 2: FULL-TEXT SEARCH (BM25)
-- ============================================================

CREATE VIRTUAL TABLE abstracts USING fts5(
    paper_id UNINDEXED,
    domain,
    paper_name,
    collection,
    title,
    body,
    tokenize='porter unicode61'
);

CREATE VIRTUAL TABLE sections USING fts5(
    paper_id UNINDEXED,
    domain,
    paper_name,
    section_title,
    section_level,
    body,
    tokenize='unicode61 remove_diacritics 2'
);

CREATE VIRTUAL TABLE terms_fts USING fts5(
    term_id UNINDEXED,
    domain,
    term_type,
    entity_type,
    canonical_name,
    aliases,
    tokenize='porter unicode61'
);

-- ============================================================
-- GROUP 3: TAXONOMY & RESOLUTION
-- ============================================================

CREATE TABLE IF NOT EXISTS canonical_terms (
    id INTEGER PRIMARY KEY,
    domain TEXT NOT NULL,
    term_type TEXT NOT NULL,
    -- For term_type='entity', entity_type (dataset/model/metric/...) is
    -- metadata on the canonical — NOT part of its identity. GLiNER2's label
    -- output is noisy within a paper (the same string gets labelled as
    -- method/software/benchmark across mentions), so keying by entity_type
    -- fragments every popular entity into 3-5 rows. entity_type holds the
    -- currently-stored label; entity_type_score holds the GLiNER2 confidence
    -- that established it. A later paper resolving the same canonical with
    -- a DIFFERENT label AND a strictly higher score overturns both fields
    -- (see resolver._maybe_flip_entity_type). This turns entity_type from
    -- frozen-on-first-insert into mutable-under-evidence so early-paper
    -- mislabels can be corrected by more confident later extractions.
    -- Non-entity rows (collections/topics) carry entity_type='' and
    -- entity_type_score=0.0; the flip path is gated on non-empty new type.
    entity_type TEXT NOT NULL DEFAULT '',
    entity_type_score REAL NOT NULL DEFAULT 0.0,
    canonical_name TEXT NOT NULL,
    first_seen_in TEXT NOT NULL,
    UNIQUE(domain, term_type, canonical_name)
);

CREATE INDEX IF NOT EXISTS idx_terms_domain_type ON canonical_terms(domain, term_type);
CREATE INDEX IF NOT EXISTS idx_terms_name ON canonical_terms(canonical_name);

CREATE TABLE IF NOT EXISTS term_aliases (
    term_id INTEGER NOT NULL REFERENCES canonical_terms(id),
    alias TEXT NOT NULL,
    source_paper TEXT NOT NULL,
    match_tier INTEGER,
    PRIMARY KEY(term_id, alias, source_paper)
);

CREATE VIRTUAL TABLE term_embeddings USING vec0(
    term_id INTEGER PRIMARY KEY,
    embedding float[384],
    term_type TEXT,
    entity_type TEXT,
    domain TEXT
);

CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY,
    paper_id INTEGER NOT NULL REFERENCES papers(id),
    domain TEXT NOT NULL,
    paper_name TEXT NOT NULL,
    entity_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    source_breadcrumb TEXT NOT NULL,
    description TEXT,
    UNIQUE(paper_id, entity_name, source_breadcrumb)
);

CREATE INDEX IF NOT EXISTS idx_entities_domain ON entities(domain);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(entity_name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(domain, entity_type);

-- ============================================================
-- GROUP 4: PAPER-TOPIC MAPPING
-- ============================================================

CREATE TABLE IF NOT EXISTS paper_topics (
    paper_id INTEGER NOT NULL REFERENCES papers(id),
    domain TEXT NOT NULL,
    topic TEXT NOT NULL,
    PRIMARY KEY(paper_id, topic)
);

-- ============================================================
-- GROUP 5: ONE-SHOT BACKFILL
-- ============================================================
-- Register collections already implied by classified papers so the
-- first-class `collections` table matches reality on any database that
-- predates it. Idempotent: INSERT OR IGNORE is a no-op once each
-- (domain, collection) pair exists. Descriptions land NULL for these
-- legacy rows — classify_paper fills them in when the LLM proposes new.
INSERT OR IGNORE INTO collections (domain, name, description)
SELECT DISTINCT domain, collection, NULL
  FROM papers
 WHERE domain IS NOT NULL AND collection IS NOT NULL;
