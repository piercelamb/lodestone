-- ============================================================
-- GROUP 1: CORE CONTENT
-- ============================================================

CREATE TABLE IF NOT EXISTS domains (
    name TEXT PRIMARY KEY,
    description TEXT
);

-- Catalog of curated collection definitions. One row per
-- (domain, collection) pair; descriptions are populated by classify_*
-- when the LLM proposes a new collection. Polymorphic `collections`
-- junction rows FK back here.
CREATE TABLE IF NOT EXISTS collection_definitions (
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

-- Polymorphic collection-membership junction. One row per (target,
-- collection) pair across papers, posts, and repos. ``target_kind``
-- discriminates; ``target_id`` references the parent row in the matching
-- table. Carries one PRIMARY collection plus 0..3 secondaries per
-- target, all within the target's single domain. Denormalized
-- ``papers.collection`` / ``posts.collection`` / ``repos.collection``
-- scalars mirror the primary row.
CREATE TABLE IF NOT EXISTS collections (
    target_kind TEXT NOT NULL,
    target_id   INTEGER NOT NULL,
    domain      TEXT NOT NULL,
    collection  TEXT NOT NULL,
    is_primary  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (target_kind, target_id, collection),
    FOREIGN KEY (domain, collection) REFERENCES collection_definitions(domain, name)
);

CREATE INDEX IF NOT EXISTS idx_collections_lookup
  ON collections(domain, collection);

-- Exactly one primary per target.
CREATE UNIQUE INDEX IF NOT EXISTS idx_collections_primary
  ON collections(target_kind, target_id) WHERE is_primary = 1;

-- Intra-domain invariant: collections.domain must match the parent
-- table's domain. The CASE arm dispatches on target_kind to read the
-- right parent table.
CREATE TRIGGER IF NOT EXISTS collections_intra_domain_insert
BEFORE INSERT ON collections
FOR EACH ROW
WHEN NEW.domain != (CASE NEW.target_kind
        WHEN 'paper' THEN (SELECT domain FROM papers WHERE id = NEW.target_id)
        WHEN 'post'  THEN (SELECT domain FROM posts  WHERE id = NEW.target_id)
        WHEN 'repo'  THEN (SELECT domain FROM repos  WHERE id = NEW.target_id)
     END)
BEGIN
    SELECT RAISE(ABORT, 'collections invariant: domain must match parent.domain');
END;

CREATE TRIGGER IF NOT EXISTS collections_intra_domain_update
BEFORE UPDATE ON collections
FOR EACH ROW
WHEN NEW.domain != (CASE NEW.target_kind
        WHEN 'paper' THEN (SELECT domain FROM papers WHERE id = NEW.target_id)
        WHEN 'post'  THEN (SELECT domain FROM posts  WHERE id = NEW.target_id)
        WHEN 'repo'  THEN (SELECT domain FROM repos  WHERE id = NEW.target_id)
     END)
BEGIN
    SELECT RAISE(ABORT, 'collections invariant: domain must match parent.domain');
END;

-- ============================================================
-- GROUP 4b: BLOG POSTS
-- ============================================================
-- Blog posts are siblings of papers — separate table, shared slug
-- namespace (`papers.paper_name` and `posts.post_name` form a global set
-- so downstream tables that key on a slug TEXT column — `sections`,
-- `term_aliases`, `topics` — work uniformly across kinds without a
-- discriminator. Tables that hard-FK to `papers.id` (e.g.
-- `post_references`) get sibling tables here for the post case;
-- collection bookkeeping lives in the polymorphic `collections` junction
-- above.

CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY,
    post_name TEXT UNIQUE NOT NULL,
    source_url TEXT NOT NULL,
    canonical_url TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    author TEXT,
    site_name TEXT,
    date TEXT NOT NULL,
    abstract TEXT NOT NULL,
    domain TEXT REFERENCES domains(name),
    collection TEXT,
    content_hash TEXT,
    etag TEXT,
    last_modified TEXT,
    raw_html TEXT,
    markdown TEXT,
    ingested_at TEXT NOT NULL,
    status TEXT NOT NULL,
    section_count INTEGER DEFAULT 0,
    entity_count INTEGER DEFAULT 0,
    needs_review INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_posts_domain ON posts(domain);
CREATE INDEX IF NOT EXISTS idx_posts_collection ON posts(domain, collection);
CREATE INDEX IF NOT EXISTS idx_posts_hash ON posts(content_hash);
CREATE INDEX IF NOT EXISTS idx_posts_review ON posts(needs_review) WHERE needs_review = 1;

-- Mirror the papers invariant: classified-or-later posts must have both
-- domain and collection. Pre-classify (FETCHED/CONVERTED) and terminal-
-- failure rows (FAILED_FETCH / FAILED_PARSE) are exempt.
CREATE TRIGGER IF NOT EXISTS posts_invariant_classified_has_domain_collection_insert
BEFORE INSERT ON posts
FOR EACH ROW
WHEN NEW.status NOT IN ('fetched', 'converted', 'failed_fetch', 'failed_parse')
 AND (NEW.domain IS NULL OR NEW.collection IS NULL)
BEGIN
    SELECT RAISE(ABORT, 'posts invariant violated: classified+ rows must have both domain and collection set');
END;

CREATE TRIGGER IF NOT EXISTS posts_invariant_classified_has_domain_collection_update
BEFORE UPDATE ON posts
FOR EACH ROW
WHEN NEW.status NOT IN ('fetched', 'converted', 'failed_fetch', 'failed_parse')
 AND (NEW.domain IS NULL OR NEW.collection IS NULL)
BEGIN
    SELECT RAISE(ABORT, 'posts invariant violated: classified+ rows must have both domain and collection set');
END;

-- Outbound arxiv citations from a post. Mirrors `paper_references` but
-- without `bibitem_id` / `ref_number` (blogs don't have a numbered
-- bibliography). `raw_text` is the link anchor text or surrounding
-- sentence — useful for surfacing the citation in search results.
CREATE TABLE IF NOT EXISTS post_references (
    id INTEGER PRIMARY KEY,
    post_id INTEGER NOT NULL REFERENCES posts(id),
    cited_arxiv_id TEXT,
    cited_paper_id INTEGER REFERENCES papers(id),
    raw_text TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_post_refs_post ON post_references(post_id);
CREATE INDEX IF NOT EXISTS idx_post_refs_cited_arxiv ON post_references(cited_arxiv_id);
CREATE INDEX IF NOT EXISTS idx_post_refs_cited_paper ON post_references(cited_paper_id);

-- ============================================================
-- GROUP 5: ONE-SHOT BACKFILL
-- ============================================================
-- Register collection definitions already implied by classified
-- sources (papers + posts + repos), then seed the polymorphic
-- `collections` junction with primary rows from the denormalized scalar
-- pointers. Idempotent: INSERT OR IGNORE is a no-op once each row
-- exists. Descriptions land NULL for legacy rows — classify_* fills them
-- in when the LLM proposes new collections. This also serves as the
-- safety net if a write path inserts a junction row before populating
-- the catalog.
INSERT OR IGNORE INTO collection_definitions (domain, name, description)
SELECT DISTINCT domain, collection, NULL
  FROM papers
 WHERE domain IS NOT NULL AND collection IS NOT NULL
UNION
SELECT DISTINCT domain, collection, NULL
  FROM posts
 WHERE domain IS NOT NULL AND collection IS NOT NULL
UNION
SELECT DISTINCT domain, collection, NULL
  FROM repos
 WHERE domain IS NOT NULL AND collection IS NOT NULL;

-- Backfill polymorphic `collections` from the per-table scalar
-- pointers so DBs that predate the polymorphic junction get a primary
-- row per classified source.
INSERT OR IGNORE INTO collections (target_kind, target_id, domain, collection, is_primary)
SELECT 'paper', id, domain, collection, 1
  FROM papers
 WHERE domain IS NOT NULL AND collection IS NOT NULL
UNION ALL
SELECT 'post', id, domain, collection, 1
  FROM posts
 WHERE domain IS NOT NULL AND collection IS NOT NULL
UNION ALL
SELECT 'repo', id, domain, collection, 1
  FROM repos
 WHERE domain IS NOT NULL AND collection IS NOT NULL;
