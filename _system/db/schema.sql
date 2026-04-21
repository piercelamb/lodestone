-- ============================================================
-- GROUP 1: CORE CONTENT
-- ============================================================

CREATE TABLE IF NOT EXISTS domains (
    name TEXT PRIMARY KEY,
    description TEXT
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
    entity_type TEXT NOT NULL DEFAULT '',
    canonical_name TEXT NOT NULL,
    first_seen_in TEXT NOT NULL,
    UNIQUE(domain, term_type, entity_type, canonical_name)
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
