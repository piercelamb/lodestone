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

-- Many-to-(few) join: a paper carries one PRIMARY collection plus 0..N
-- secondary collections, all within the paper's single domain. The
-- denormalized `papers.collection` scalar mirrors the primary row.
CREATE TABLE IF NOT EXISTS paper_collections (
    paper_id   INTEGER NOT NULL REFERENCES papers(id),
    domain     TEXT    NOT NULL,
    collection TEXT    NOT NULL,
    is_primary INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (paper_id, collection),
    FOREIGN KEY (domain, collection) REFERENCES collections(domain, name)
);

CREATE INDEX IF NOT EXISTS idx_paper_collections_collection
  ON paper_collections(domain, collection);

-- Exactly one primary per paper.
CREATE UNIQUE INDEX IF NOT EXISTS idx_paper_collections_primary
  ON paper_collections(paper_id) WHERE is_primary = 1;

-- Intra-domain invariant: paper_collections.domain must match papers.domain.
CREATE TRIGGER IF NOT EXISTS paper_collections_intra_domain_insert
BEFORE INSERT ON paper_collections
FOR EACH ROW
WHEN NEW.domain != (SELECT domain FROM papers WHERE id = NEW.paper_id)
BEGIN
    SELECT RAISE(ABORT, 'paper_collections invariant: domain must match papers.domain');
END;

CREATE TRIGGER IF NOT EXISTS paper_collections_intra_domain_update
BEFORE UPDATE ON paper_collections
FOR EACH ROW
WHEN NEW.domain != (SELECT domain FROM papers WHERE id = NEW.paper_id)
BEGIN
    SELECT RAISE(ABORT, 'paper_collections invariant: domain must match papers.domain');
END;

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

-- Invariant: every classified-or-later paper must have BOTH a domain
-- and a collection. classify_paper enforces this on the writeable side;
-- these triggers are the database-level safety net so direct SQL writes
-- (manual fixes, future migrations, third-party tools) cannot violate
-- the invariant either. Pre-classify and terminal-failure rows are
-- exempt — they don't yet have, or never will get, a domain/collection.
CREATE TRIGGER IF NOT EXISTS papers_invariant_classified_has_domain_collection_insert
BEFORE INSERT ON papers
FOR EACH ROW
WHEN NEW.status NOT IN ('fetched', 'converted', 'failed_html')
 AND (NEW.domain IS NULL OR NEW.collection IS NULL)
BEGIN
    SELECT RAISE(ABORT, 'papers invariant violated: classified+ rows must have both domain and collection set');
END;

CREATE TRIGGER IF NOT EXISTS papers_invariant_classified_has_domain_collection_update
BEFORE UPDATE ON papers
FOR EACH ROW
WHEN NEW.status NOT IN ('fetched', 'converted', 'failed_html')
 AND (NEW.domain IS NULL OR NEW.collection IS NULL)
BEGIN
    SELECT RAISE(ABORT, 'papers invariant violated: classified+ rows must have both domain and collection set');
END;

-- First-class repo entity. Addressed by ``repo_slug`` (analog of
-- ``papers.paper_name``). ``paper_id`` is NULL for standalone repos and
-- set for paper-linked repos. Standalone repos either reach CLASSIFIED
-- (README-driven) or ORPHANED (no usable README); paper-linked repos
-- reach REPO_FETCHED and inherit domain/collection from the paper.
CREATE TABLE IF NOT EXISTS repos (
    id INTEGER PRIMARY KEY,
    repo_slug TEXT UNIQUE NOT NULL,
    url TEXT UNIQUE NOT NULL,
    host TEXT NOT NULL,
    owner TEXT NOT NULL,
    name TEXT NOT NULL,
    paper_id INTEGER REFERENCES papers(id),
    description TEXT,
    default_branch TEXT,
    commit_sha TEXT,
    fetched_at TEXT,
    ingested_at TEXT NOT NULL,
    domain TEXT REFERENCES domains(name),
    collection TEXT,
    status TEXT NOT NULL,
    needs_review INTEGER NOT NULL DEFAULT 0,
    file_count INTEGER DEFAULT 0,
    has_readme INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_repos_paper ON repos(paper_id);
CREATE INDEX IF NOT EXISTS idx_repos_domain ON repos(domain);
CREATE INDEX IF NOT EXISTS idx_repos_collection ON repos(domain, collection);

-- Mirror the papers invariant: classified-or-later repos must have both
-- domain and collection. ORPHANED, FAILED_*, and pre-classify rows are
-- exempt. RESOLVED + REPO_FETCHED are pre-classify; for paper-linked
-- repos those stages already carry inherited domain/collection so the
-- trigger is also satisfied if classification is implicitly skipped.
CREATE TRIGGER IF NOT EXISTS repos_invariant_classified_has_domain_collection_insert
BEFORE INSERT ON repos
FOR EACH ROW
WHEN NEW.status NOT IN ('resolved', 'repo_fetched', 'orphaned', 'failed_resolve', 'failed_repo')
 AND (NEW.domain IS NULL OR NEW.collection IS NULL)
BEGIN
    SELECT RAISE(ABORT, 'repos invariant violated: classified+ rows must have both domain and collection set');
END;

CREATE TRIGGER IF NOT EXISTS repos_invariant_classified_has_domain_collection_update
BEFORE UPDATE ON repos
FOR EACH ROW
WHEN NEW.status NOT IN ('resolved', 'repo_fetched', 'orphaned', 'failed_resolve', 'failed_repo')
 AND (NEW.domain IS NULL OR NEW.collection IS NULL)
BEGIN
    SELECT RAISE(ABORT, 'repos invariant violated: classified+ rows must have both domain and collection set');
END;

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

-- Drop the legacy page_images table on any DB that predates the removal.
-- LaTeXML strips page layout, so chunks have no page index to correlate
-- back to a render — the table was vestigial. Idempotent.
DROP TABLE IF EXISTS page_images;

-- Drop the legacy `abstracts` FTS5 virtual table on any DB that predates
-- the removal. The abstract text is fully indexed by `sections` (every
-- paper's `# Abstract` section is one row, plus the level-1 markdown
-- blob row contains it as a substring). Paper-level rollups happen via
-- `GROUP BY paper_name` on `sections`. The structured `papers.abstract`
-- column stays for display / export / future non-BM25 use. Dropping the
-- virtual table also drops its shadow tables (abstracts_data,
-- abstracts_idx, abstracts_content, abstracts_docsize, abstracts_config).
DROP TABLE IF EXISTS abstracts;

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

CREATE VIRTUAL TABLE sections USING fts5(
    paper_id UNINDEXED,
    domain,
    paper_name,
    section_title,
    section_level,
    body,
    tokenize='unicode61 remove_diacritics 2'
);

-- One row per source file kept under a repo. Navigated by path
-- (``--repo-tree REPO`` lists all rows; ``--read-code REPO --path X``
-- fetches one). Binary blobs / vendored deps / model weights are
-- filtered out by ``fetch_repo.py`` before insert; surviving content is
-- guaranteed UTF-8 decodable text. ``.ipynb`` files are flattened to
-- code+markdown cell text and stored under their original path.
CREATE TABLE IF NOT EXISTS code_files (
    id INTEGER PRIMARY KEY,
    repo_id INTEGER NOT NULL REFERENCES repos(id),
    path TEXT NOT NULL,
    language TEXT,
    size_bytes INTEGER NOT NULL,
    content TEXT NOT NULL,
    UNIQUE(repo_id, path)
);
CREATE INDEX IF NOT EXISTS idx_code_files_repo ON code_files(repo_id);

-- Parallel FTS5 surface over each repo's top-level README. One row per
-- repo at most. README content is also present in ``code_files`` under
-- its real path; ``readmes_fts`` is the searchable copy. Tokenizer
-- matches ``sections`` so ``--scope both`` query behavior is uniform.
-- ``repo_slug`` is the addressable id; ``domain`` is the repo's domain
-- (NULL for orphans).
CREATE VIRTUAL TABLE readmes_fts USING fts5(
    repo_id UNINDEXED,
    repo_slug UNINDEXED,
    domain,
    path,
    content,
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

-- term_aliases is a per-(concept, paper) synonym index — one row per
-- non-canonical surface form per paper. Tier-1 hits (alias equals the
-- canonical) and tier-5 mints (the new canonical itself) write nothing;
-- only tier-2/3/4 hits whose normalized form differs from the canonical's
-- normalized form land here. Invariant: `alias != canonical_name` for
-- every row's `term_id` after normalization. Old DBs with the 3-column
-- (term_id, alias, source_paper) PK + sibling `entities` table, and the
-- short-lived 4-column appearance-log shape with `source_breadcrumb`,
-- are both migrated by `_system.db.migrations._migrate_entities_to_aliases`.
CREATE TABLE IF NOT EXISTS term_aliases (
    term_id INTEGER NOT NULL REFERENCES canonical_terms(id),
    alias TEXT NOT NULL,
    source_paper TEXT NOT NULL,
    match_tier INTEGER,
    PRIMARY KEY(term_id, alias, source_paper)
);

CREATE INDEX IF NOT EXISTS idx_aliases_source_paper ON term_aliases(source_paper);

CREATE VIRTUAL TABLE term_embeddings USING vec0(
    term_id INTEGER PRIMARY KEY,
    embedding float[384],
    term_type TEXT,
    entity_type TEXT,
    domain TEXT
);

-- ============================================================
-- GROUP 4: TOPIC MAPPING (papers + repos)
-- ============================================================

-- Unified topic table for both paper and repo classification. The
-- ``target_kind`` column discriminates; ``target_id`` references either
-- ``papers.id`` or ``repos.id``. The PK ensures one (target, topic)
-- pair regardless of kind.
CREATE TABLE IF NOT EXISTS topics (
    target_kind TEXT NOT NULL,
    target_id INTEGER NOT NULL,
    domain TEXT NOT NULL,
    topic TEXT NOT NULL,
    PRIMARY KEY (target_kind, target_id, topic)
);

CREATE INDEX IF NOT EXISTS idx_topics_target ON topics(target_kind, target_id);
CREATE INDEX IF NOT EXISTS idx_topics_topic ON topics(domain, topic);

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

-- Backfill paper_collections from the legacy scalar `papers.collection`
-- so DBs that predate the join table get a primary row per classified paper.
INSERT OR IGNORE INTO paper_collections (paper_id, domain, collection, is_primary)
SELECT id, domain, collection, 1
  FROM papers
 WHERE domain IS NOT NULL AND collection IS NOT NULL;
