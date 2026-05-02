"""Diagnostic: measure taxonomy-kind diversity in `mode_search` results.

Runs a default set of representative queries (proper-noun, broad concept,
narrow phrase, ambiguous, out-of-corpus) against the real ``lodestone.db``
and tallies per-kind hit counts in the ``taxonomy`` bucket. The output is
the input to deciding whether the simple "single FTS + KNN fallback" path
is good enough or whether we need a per-kind cap / round-robin merge to
prevent one kind from monopolising top-k slots.

Usage::

    uv run python -m _system.scripts.eval_search_diversity
    uv run python -m _system.scripts.eval_search_diversity --limit 5 \
        --queries "ToT" "Game of 24" "tree search"

Reads ``LODESTONE_DB`` env var (or ``--db`` flag) for the database path.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from _system.db.connection import get_conn
from _system.scripts.search import mode_search


# Default query set — chosen to span shapes likely to hit different kinds.
# With a 1-paper Tree-of-Thoughts corpus, the breakdown leans toward ToT
# vocabulary; swap with --queries when running against richer corpora.
_DEFAULT_QUERIES: list[str] = [
    # Proper-noun / known entity surface forms
    "ToT",
    "GPT-4",
    "Game of 24",
    # Method-flavoured (entity-typed methods)
    "tree search",
    "breadth-first search",
    # Topic-flavoured phrases (long-form deliberate-reasoning topics)
    "deliberate reasoning",
    "intermediate reasoning",
    # Collection-flavoured
    "Deliberate Search over Thought Structures",
    # Broad concepts likely to span kinds
    "reasoning",
    "evaluation",
    # Out-of-corpus — should mostly miss or pull weak vector neighbors
    "diffusion model image generation",
]


def _resolve_db(arg: str | None) -> Path:
    if arg:
        return Path(arg).expanduser().resolve()
    env = os.environ.get("LODESTONE_DB")
    if env:
        return Path(env).expanduser().resolve()
    here = Path.cwd()
    db = here / "lodestone.db"
    if db.is_file():
        return db
    raise SystemExit(
        "could not resolve database; pass --db or set LODESTONE_DB"
    )


def _format_row(row: dict, max_name: int = 38) -> str:
    name = row["canonical_name"]
    if len(name) > max_name:
        name = name[: max_name - 1] + "…"
    et = row.get("entity_type") or ""
    et_disp = f" [{et}]" if et else ""
    return f"    {row['kind']:11s} {name}{et_disp}"


def _run(queries: list[str], db_path: Path, limit: int) -> dict:
    conn = get_conn(db_path)
    try:
        per_query: list[dict] = []
        global_counter: Counter[str] = Counter()
        kind_first_slot: Counter[str] = Counter()
        for q in queries:
            payload = mode_search(conn, query=q, filters={}, limit=limit)
            taxonomy = payload.get("taxonomy", []) or []
            kinds = [r["kind"] for r in taxonomy]
            local = Counter(kinds)
            global_counter.update(local)
            if kinds:
                kind_first_slot[kinds[0]] += 1
            per_query.append({
                "query": q,
                "n_taxonomy": len(taxonomy),
                "by_kind": dict(local),
                "rows": taxonomy,
                "n_sections": len(payload.get("sections", []) or []),
                "n_readmes": len(payload.get("readmes", []) or []),
                "status": payload.get("status"),
            })
        return {
            "db": str(db_path),
            "limit": limit,
            "n_queries": len(queries),
            "per_query": per_query,
            "global_kind_counts": dict(global_counter),
            "first_slot_kind_counts": dict(kind_first_slot),
        }
    finally:
        conn.close()


def _render_report(report: dict) -> str:
    lines: list[str] = []
    lines.append(f"# Search kind-diversity eval")
    lines.append("")
    lines.append(f"db: `{report['db']}`")
    lines.append(f"limit per query: {report['limit']}")
    lines.append(f"queries: {report['n_queries']}")
    lines.append("")
    lines.append("## Aggregate")
    total_rows = sum(report["global_kind_counts"].values())
    lines.append(f"total taxonomy rows returned: {total_rows}")
    if total_rows:
        for kind, n in sorted(report["global_kind_counts"].items(),
                              key=lambda kv: -kv[1]):
            pct = 100.0 * n / total_rows
            lines.append(f"  {kind:11s} {n:4d} ({pct:5.1f}%)")
    lines.append("")
    lines.append("first-slot-kind frequency (which kind grabbed the #1 slot):")
    fst = report["first_slot_kind_counts"]
    fst_total = sum(fst.values())
    if fst_total:
        for kind, n in sorted(fst.items(), key=lambda kv: -kv[1]):
            pct = 100.0 * n / fst_total
            lines.append(f"  {kind:11s} {n:4d} ({pct:5.1f}%)")
    lines.append("")
    lines.append("## Per-query")
    for entry in report["per_query"]:
        lines.append("")
        bk = ", ".join(f"{k}={v}" for k, v in entry["by_kind"].items()) or "—"
        status = f"  status={entry['status']}" if entry["status"] else ""
        lines.append(
            f"### `{entry['query']}` "
            f"(taxonomy:{entry['n_taxonomy']}/{report['limit']}, "
            f"sections:{entry['n_sections']}, readmes:{entry['n_readmes']}){status}"
        )
        lines.append(f"  by_kind: {bk}")
        for row in entry["rows"]:
            lines.append(_format_row(row))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=None, help="path to lodestone.db")
    p.add_argument("--limit", type=int, default=5,
                   help="taxonomy bucket size per query (default 5)")
    p.add_argument("--queries", nargs="+", default=None,
                   help="override default query list")
    p.add_argument("--json", action="store_true",
                   help="emit raw JSON instead of human report")
    args = p.parse_args(argv)

    db_path = _resolve_db(args.db)
    queries = args.queries or _DEFAULT_QUERIES
    report = _run(queries, db_path, args.limit)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(_render_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
