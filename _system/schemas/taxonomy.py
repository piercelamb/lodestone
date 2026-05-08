"""Taxonomy/classification payload schemas.

Two-stage shape:

- :class:`ClassificationLLMOutput` — the raw structured payload the LLM
  returns. Domain is a zero-based *index* into the runtime-provided
  ``existing_domains`` list (``-1`` means "propose a new domain"); the
  proposed name lives in a sibling string field. Collections are an
  ordered list of :class:`CollectionPick` entries (primary at index 0,
  secondaries 1+, all within the chosen domain). This shape is strict-
  mode compatible across Anthropic tool_use, OpenAI json_schema, and
  Gemini responseSchema.
- :class:`ClassificationOutput` — the resolved shape produced inside
  ``classify_paper`` after the index→name lookup. Downstream resolver and
  DB writes operate on this. ``collections[0]`` is always the paper's
  primary; secondaries follow.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class CollectionPick(BaseModel):
    """One pick within the LLM's ``collections`` list.

    ``index`` is either a zero-based index into the chosen domain's
    collection list (as shown beneath that domain in the rendered tree)
    or ``-1`` to propose a new collection. When ``-1``, ``new_name``
    carries the proposed collection name and ``new_desc`` carries a
    short sentence describing the cluster of work; both must be empty
    when ``index >= 0``.
    """

    model_config = ConfigDict(extra="forbid")

    index: int
    new_name: str
    new_desc: str


class ClassificationLLMOutput(BaseModel):
    """Raw structured output from the classify_paper LLM call.

    Domain uses the "index + proposed name" split:

    - ``domain_index`` is the zero-based index into the runtime
      ``existing_domains`` list, or ``-1`` to propose a new domain. When
      ``-1``, ``new_domain`` carries the new name and
      ``new_domain_desc`` carries a short human-readable
      sentence that will be stored alongside the domain row.

    Collections are a list of :class:`CollectionPick` entries, ordered
    primary-first (index 0) followed by 0..3 secondary memberships, all
    within the chosen domain. Each entry can be either an existing
    index or ``-1`` (propose new). Min 1, max 4 entries.

    Post-validation in :func:`classify_paper._resolve_raw` enforces the
    cross-field rules the strict-mode enum cannot (each pick's index
    must be in range for the chosen domain; a new domain forces every
    pick to be ``index == -1``; ``new_domain_desc`` must be non-empty
    iff ``domain_index == -1``; per-pick ``new_name``/``new_desc`` must
    be non-empty iff that pick's ``index == -1``).
    """

    model_config = ConfigDict(extra="forbid")

    domain_index: int
    new_domain: str
    new_domain_desc: str
    collections: list[CollectionPick]
    topics: list[str]


class ResolvedCollection(BaseModel):
    """One collection on a classified paper, post-resolution.

    ``description`` holds the short description the LLM supplied for a
    newly-proposed collection; ``None`` when the LLM picked an existing
    entry. The primary is ``collections[0]``; secondaries follow.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None


class ClassificationOutput(BaseModel):
    """Resolved classification, post index→name lookup.

    ``domain_description`` holds the short description the LLM supplied
    for a newly-proposed domain; ``None`` when the LLM picked an
    existing entry (no description to write) or when ``--domain-override``
    forced a choice without LLM input (operator can fill it in later).
    ``collections`` is the ordered list of resolved collections; index 0
    is the primary.
    """

    model_config = ConfigDict(extra="forbid")

    domain: str
    domain_is_new: bool
    domain_description: str | None
    collections: list[ResolvedCollection]
    topics: list[str]


