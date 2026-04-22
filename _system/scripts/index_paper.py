"""Populate FTS5 + vec0 derived tables from the authoritative source tables.

Fifth (and final) pipeline stage: after ``extract_entities.py`` has written
canonical entity rows, this script walks ``papers``/``entities``/``paper_topics``
and builds the three FTS5 tables (``abstracts``, ``sections``, ``terms_fts``)
plus the vec0 ``term_embeddings`` table that the search layer reads.

Two modes:

- ``index_one(paper_name=...)`` — per-paper replace-all for a single row.
  Advances ``papers.status`` from ``EXTRACTED`` to ``INDEXED``. The
  ``terms_fts`` rebuild inside ``index_one`` is **scoped** to the terms this
  paper touches: its entity canonicals, its ``paper_topics`` canonicals, and
  its ``collection`` canonical. Indexing paper A MUST NOT touch term rows
  whose only producer is paper B.

- ``rebuild_all()`` — offline full rebuild. Drops and recreates all four
  derived tables, then re-populates every paper's ``abstracts``/``sections``
  rows, every canonical term's ``terms_fts`` row, and every canonical term's
  ``term_embeddings`` vector (via ``Embedder.embed_batch`` in windows of 64).
  Intended for schema / corpus migrations; corruption recovery path. Not
  concurrency-safe with live search queries.

Invariants worth keeping straight:

- ``abstracts`` / ``sections`` FTS5 tables have no FK to ``papers``; the
  ``DELETE FROM <tbl> WHERE paper_name = ?`` calls in ``index_one`` are
  load-bearing for re-index idempotency.
- ``sections.body`` = the ``SectionChunk.body`` returned by
  :func:`_system.utils.sections.split_sections`, which already contains the
  breadcrumb prefix on line 1. That is the reason a BM25 query for a parent
  section's token hits a child chunk.
- ``terms_fts.aliases`` is a single space-joined string of ``DISTINCT`` alias
  surface forms from ``term_aliases``. The porter tokenizer on ``terms_fts``
  stems each token for NL-query matching.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import NamedTuple

import sqlite_vec

from _system.db.connection import get_conn, transaction
from _system.db.migrations import VIRTUAL_TABLE_BLOCK_RE
from _system.resolution.embeddings import Embedder
from _system.schemas.paper_metadata import PaperStatus, can_run_from
from _system.utils.logging import get_logger
from _system.utils.sections import split_sections

_LOG = get_logger("scripts.index_paper")

_EMBED_BATCH_SIZE = 64
_PAPERS_BATCH_SIZE = 500
_TERMS_BATCH_SIZE = 500

_SCHEMA_FILE = Path(__file__).resolve().parent.parent / "db" / "schema.sql"

_DERIVED_VIRTUAL_TABLES: tuple[str, ...] = (
    "abstracts",
    "sections",
    "terms_fts",
    "term_embeddings",
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class IndexPaperError(Exception):
    """Base class for index_paper failures."""


class PaperNotFound(IndexPaperError):
    """No papers row for the given paper_name."""


class StatusTooLow(IndexPaperError):
    """can_run_from rejected current status for INDEXED (and --force not set)."""


class UnknownStatusError(IndexPaperError):
    """papers.status holds a string not recognized by PaperStatus."""


class IndexResult(NamedTuple):
    paper_name: str
    section_count: int
    status: str


# ---------------------------------------------------------------------------
# Per-paper indexing
# ---------------------------------------------------------------------------


def index_one(
    *,
    paper_name: str,
    conn: sqlite3.Connection,
    force: bool = False,
) -> IndexResult:
    """Populate abstracts/sections/terms_fts for one paper; advance to INDEXED.

    Replace-all semantics: deletes any prior ``abstracts`` / ``sections`` rows
    for ``paper_name`` before inserting. The ``terms_fts`` rebuild is scoped to
    the touched-term set (entities this paper produced, topics it was tagged
    with, and the canonical row of its ``collection``). Commits in a single
    transaction so a mid-stage raise leaves ``papers.status`` unchanged.
    """
    row = conn.execute(
        """
        SELECT id, domain, collection, title, abstract, markdown, status
          FROM papers WHERE paper_name = ?
        """,
        (paper_name,),
    ).fetchone()
    if row is None:
        raise PaperNotFound(f"paper_name={paper_name!r} not found in papers table")
    paper_id, domain, collection, title, abstract, markdown, status_str = row

    try:
        current = PaperStatus(status_str) if status_str else None
    except ValueError as exc:
        raise UnknownStatusError(
            f"paper_name={paper_name!r}: unrecognized status={status_str!r}"
        ) from exc

    if not force and not can_run_from(current, PaperStatus.INDEXED):
        extra = (
            " (FAILED_HTML is terminal — re-fetch required)"
            if current is PaperStatus.FAILED_HTML
            else ""
        )
        raise StatusTooLow(
            f"paper_name={paper_name!r}: cannot run INDEXED from status="
            f"{status_str!r}{extra}"
        )

    with transaction(conn):
        # Rerun cleanup. FTS5 tables don't honor FK CASCADE — skipping these
        # produces duplicate hits on re-index.
        conn.execute("DELETE FROM abstracts WHERE paper_name = ?", (paper_name,))
        conn.execute("DELETE FROM sections WHERE paper_name = ?", (paper_name,))

        conn.execute(
            """
            INSERT INTO abstracts
                (paper_id, domain, paper_name, collection, title, body)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (paper_id, domain, paper_name, collection, title, abstract or ""),
        )

        new_section_count = 0
        if markdown:
            new_section_count = _insert_sections_for_paper(
                conn,
                paper_id=paper_id,
                domain=domain,
                paper_name=paper_name,
                markdown=markdown,
            )

        touched = _touched_term_ids(
            conn, paper_id=paper_id, domain=domain, collection=collection,
        )
        if not touched:
            _LOG.debug(
                "paper_id=%s paper_name=%s: no touched canonical terms "
                "(no entities, topics, or collection canonical)",
                paper_id, paper_name,
            )
        _rebuild_terms_fts(conn, _fetch_canonical_rows(conn, touched))

        conn.execute(
            """
            UPDATE papers
               SET section_count = ?,
                   status = ?
             WHERE paper_name = ?
            """,
            (new_section_count, PaperStatus.INDEXED.value, paper_name),
        )

    _LOG.info(
        "indexed paper_id=%s paper_name=%s sections=%d terms_fts_touched=%d",
        paper_id, paper_name, new_section_count, len(touched),
    )
    return IndexResult(
        paper_name=paper_name,
        section_count=new_section_count,
        status=PaperStatus.INDEXED.value,
    )


def _insert_sections_for_paper(
    conn: sqlite3.Connection,
    *,
    paper_id: int,
    domain: str | None,
    paper_name: str,
    markdown: str,
) -> int:
    """Split markdown and insert one ``sections`` FTS5 row per chunk.

    ``SectionChunk.body`` already has the breadcrumb prepended
    (``{breadcrumb}\\n\\n{raw_body}``), so we use it verbatim — prepending it
    again would duplicate the breadcrumb line and dilute BM25 scoring.

    Returns the number of section rows inserted.
    """
    rows = [
        (paper_id, domain, paper_name, chunk.title, str(chunk.level), chunk.body)
        for chunk in split_sections(markdown)
    ]
    if rows:
        conn.executemany(
            """
            INSERT INTO sections
                (paper_id, domain, paper_name, section_title,
                 section_level, body)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return len(rows)


def _touched_term_ids(
    conn: sqlite3.Connection,
    *,
    paper_id: int,
    domain: str | None,
    collection: str | None,
) -> set[int]:
    """Union of canonical term ids this paper touches.

    Three sources:
    - ``entities`` rows joined back to ``canonical_terms`` on
      ``(domain, 'entity', entity_type, entity_name)``.
    - ``paper_topics`` rows joined to ``canonical_terms`` on
      ``(domain, 'topic', '', topic)``.
    - ``papers.collection`` matched against ``canonical_terms`` on
      ``(domain, 'collection', '', collection)``.

    Entities / paper_topics carry the canonical *name* (not term_id) in
    schema, so we re-join to canonical_terms here. A matching row is required;
    an entity whose canonical row has been deleted is silently dropped from
    the touched set (the orphan itself is a separate data-integrity problem).
    """
    term_ids: set[int] = set()

    # LEFT JOIN so orphan entity_names (no matching canonical_terms row) show
    # up as NULL ct.id. We count them to surface a WARN if extract_entities'
    # contract of writing canonical names into entities.entity_name breaks.
    rows = conn.execute(
        """
        SELECT DISTINCT e.entity_name, ct.id
          FROM entities e
          LEFT JOIN canonical_terms ct
            ON ct.domain = e.domain
           AND ct.term_type = 'entity'
           AND ct.entity_type = e.entity_type
           AND ct.canonical_name = e.entity_name
         WHERE e.paper_id = ?
        """,
        (paper_id,),
    ).fetchall()
    matched_ids = {r[1] for r in rows if r[1] is not None}
    term_ids.update(matched_ids)
    orphan_names = {r[0] for r in rows if r[1] is None}
    if orphan_names:
        _LOG.warning(
            "paper_id=%s: %d distinct entity_names in entities have no "
            "matching canonical_terms row — check extract_entities invariants.",
            paper_id, len(orphan_names),
        )

    rows = conn.execute(
        """
        SELECT DISTINCT ct.id
          FROM paper_topics pt
          JOIN canonical_terms ct
            ON ct.domain = pt.domain
           AND ct.term_type = 'topic'
           AND ct.entity_type = ''
           AND ct.canonical_name = pt.topic
         WHERE pt.paper_id = ?
        """,
        (paper_id,),
    ).fetchall()
    term_ids.update(r[0] for r in rows)

    # domain / collection can be None (pre-classify) or "" (sentinel from
    # extract_entities when classify didn't run). Both are falsy; skip the
    # collection lookup in either case.
    if domain and collection:
        row = conn.execute(
            """
            SELECT id FROM canonical_terms
             WHERE domain = ? AND term_type = 'collection'
               AND entity_type = '' AND canonical_name = ?
            """,
            (domain, collection),
        ).fetchone()
        if row is not None:
            term_ids.add(row[0])

    return term_ids


def _fetch_canonical_rows(
    conn: sqlite3.Connection, term_ids: set[int],
) -> list[tuple[int, str, str, str, str]]:
    """Fetch ``(id, domain, term_type, entity_type, canonical_name)`` rows
    for ``term_ids`` in a single query. An id whose row was concurrently
    deleted is silently dropped from the result."""
    if not term_ids:
        return []
    placeholders = ",".join("?" * len(term_ids))
    return conn.execute(
        f"""
        SELECT id, domain, term_type, entity_type, canonical_name
          FROM canonical_terms WHERE id IN ({placeholders})
        """,
        list(term_ids),
    ).fetchall()


def _rebuild_terms_fts(
    conn: sqlite3.Connection,
    canonical_rows: list[tuple[int, str, str, str, str]],
) -> None:
    """DELETE + INSERT ``terms_fts`` rows for the given canonical rows.

    Each row is ``(term_id, domain, term_type, entity_type, canonical_name)``.
    Aliases are fetched with a single ``IN`` query, space-joined per term,
    then written via ``executemany``.

    ``term_id`` is UNINDEXED in the virtual table, so DELETE is a linear
    scan — acceptable at scoped-per-paper scale. ``rebuild_all`` runs this
    after DROP+CREATE, so the DELETE is a no-op there.
    """
    if not canonical_rows:
        return
    term_ids = [r[0] for r in canonical_rows]
    placeholders = ",".join("?" * len(term_ids))

    aliases_by_term: dict[int, list[str]] = {}
    for tid, alias in conn.execute(
        f"SELECT DISTINCT term_id, alias FROM term_aliases "
        f"WHERE term_id IN ({placeholders})",
        term_ids,
    ):
        aliases_by_term.setdefault(tid, []).append(alias)

    conn.execute(
        f"DELETE FROM terms_fts WHERE term_id IN ({placeholders})",
        term_ids,
    )
    conn.executemany(
        """
        INSERT INTO terms_fts
            (term_id, domain, term_type, entity_type,
             canonical_name, aliases)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (tid, dom, t_type, ent_type, can_name,
             " ".join(aliases_by_term.get(tid, [])))
            for (tid, dom, t_type, ent_type, can_name) in canonical_rows
        ],
    )


# ---------------------------------------------------------------------------
# Full rebuild
# ---------------------------------------------------------------------------


def rebuild_all(
    conn: sqlite3.Connection,
    *,
    embedder: Embedder | None = None,
) -> None:
    """Offline rebuild of all four derived tables. See module docstring.

    ``embedder`` is a test seam — production passes None and the real
    BGE embedder is instantiated lazily (only if there are canonicals to
    embed, so an empty DB avoids the model load).
    """
    _drop_and_recreate_derived_virtual_tables(conn)

    paper_rows = conn.execute(
        """
        SELECT id, domain, paper_name, collection, title, abstract, markdown
          FROM papers ORDER BY id
        """
    ).fetchall()
    paper_count = len(paper_rows)
    for i in range(0, paper_count, _PAPERS_BATCH_SIZE):
        batch = paper_rows[i : i + _PAPERS_BATCH_SIZE]
        with transaction(conn):
            for row in batch:
                _insert_abstract_and_sections_for_row(conn, row)

    canonical_rows = conn.execute(
        """
        SELECT id, domain, term_type, entity_type, canonical_name
          FROM canonical_terms
         ORDER BY id
        """
    ).fetchall()
    for i in range(0, len(canonical_rows), _TERMS_BATCH_SIZE):
        batch = canonical_rows[i : i + _TERMS_BATCH_SIZE]
        with transaction(conn):
            _rebuild_terms_fts(conn, batch)

    if not canonical_rows:
        _LOG.info("rebuild_all: no canonical_terms rows; skipping embedding pass")
        return

    if embedder is None:
        embedder = Embedder()

    total_embedded = 0
    for i in range(0, len(canonical_rows), _EMBED_BATCH_SIZE):
        batch = canonical_rows[i : i + _EMBED_BATCH_SIZE]
        texts = [r[4] for r in batch]
        vectors = embedder.embed_batch(texts)
        with transaction(conn):
            conn.executemany(
                """
                INSERT INTO term_embeddings
                    (term_id, embedding, term_type, entity_type, domain)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        term_id,
                        sqlite_vec.serialize_float32(vec),
                        t_type,
                        ent_type,
                        dom,
                    )
                    for (term_id, dom, t_type, ent_type, _name), vec
                    in zip(batch, vectors)
                ],
            )
        total_embedded += len(batch)

    _LOG.info(
        "rebuild_all: papers=%d canonicals=%d embedded=%d",
        paper_count, len(canonical_rows), total_embedded,
    )


def _insert_abstract_and_sections_for_row(
    conn: sqlite3.Connection,
    paper_row: tuple[int, str | None, str, str | None, str | None, str | None, str | None],
) -> None:
    """Per-paper insert helper used by ``rebuild_all``.

    Skips the DELETE step (we already DROP+CREATE'd the parent virtual
    tables) and the status flip (not a status-changing operation).
    """
    paper_id, domain, paper_name, collection, title, abstract, markdown = paper_row
    conn.execute(
        """
        INSERT INTO abstracts
            (paper_id, domain, paper_name, collection, title, body)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (paper_id, domain, paper_name, collection, title, abstract or ""),
    )
    if markdown:
        _insert_sections_for_paper(
            conn,
            paper_id=paper_id,
            domain=domain,
            paper_name=paper_name,
            markdown=markdown,
        )


def _drop_and_recreate_derived_virtual_tables(conn: sqlite3.Connection) -> None:
    """DROP+CREATE the four derived virtual tables using DDL from schema.sql.

    Avoids duplicating the CREATE VIRTUAL TABLE statements in code — schema.sql
    stays the single source of truth for derived-table DDL.

    The four DROP+CREATE pairs run in a single transaction so a process crash
    or CREATE failure cannot leave the schema half-destroyed (connection.py
    runs in autocommit mode; without the wrapper each DROP would commit on its
    own).
    """
    schema_sql = _SCHEMA_FILE.read_text(encoding="utf-8")
    found: dict[str, str] = {}
    for match in VIRTUAL_TABLE_BLOCK_RE.finditer(schema_sql):
        name = match.group(1)
        found[name] = match.group(0)

    with transaction(conn):
        for name in _DERIVED_VIRTUAL_TABLES:
            stmt = found.get(name)
            if stmt is None:
                raise RuntimeError(
                    f"rebuild_all: missing CREATE VIRTUAL TABLE {name!r} in "
                    f"schema.sql — did the schema drift?"
                )
            conn.execute(f"DROP TABLE IF EXISTS {name}")
            conn.execute(stmt)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Populate FTS5 + vec0 derived tables from authoritative source "
            "tables. Per-paper mode advances papers.status to 'indexed'."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--paper", default=None, help="papers.paper_name to index")
    mode.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop and rebuild ALL derived tables from scratch (offline).",
    )
    parser.add_argument("--db", default="lodestone.db", help="sqlite db path")
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Required for --rebuild (confirms you really want to wipe and "
            "recreate the derived tables). Ignored for --paper."
        ),
    )
    args = parser.parse_args(argv)

    if args.rebuild and not args.force:
        parser.error("--rebuild requires --force to confirm wipe-and-recreate")

    conn = get_conn(Path(args.db))
    try:
        if args.rebuild:
            rebuild_all(conn)
            print(json.dumps({"mode": "rebuild", "status": "ok"}))
        else:
            result = index_one(paper_name=args.paper, conn=conn)
            print(json.dumps(result._asdict()))
    finally:
        conn.close()


if __name__ == "__main__":
    _main()
