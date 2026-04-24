"""Taxonomy/classification payload schemas.

Two-stage shape:

- :class:`ClassificationLLMOutput` — the raw structured payload the LLM
  returns. Domain is a zero-based *index* into the runtime-provided
  ``existing_domains`` list (``-1`` means "propose a new domain"); the
  proposed name lives in a sibling string field. This shape is strict-
  mode compatible across Anthropic tool_use, OpenAI json_schema, and
  Gemini responseSchema.
- :class:`ClassificationOutput` — the resolved shape produced inside
  ``classify_paper`` after the index→name lookup. Downstream resolver and
  DB writes operate on this.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ClassificationLLMOutput(BaseModel):
    """Raw structured output from the classify_paper LLM call.

    Domain and collection both use the "index + proposed name" split:

    - ``domain_index`` is the zero-based index into the runtime
      ``existing_domains`` list, or ``-1`` to propose a new domain. When
      ``-1``, ``proposed_new_domain`` carries the new name and
      ``proposed_new_domain_description`` carries a short human-readable
      sentence that will be stored alongside the domain row.
    - ``collection_index`` is the zero-based index into the chosen domain's
      collection list (as shown beneath that domain in the rendered tree),
      or ``-1`` to propose a new collection. When ``-1``,
      ``proposed_new_collection`` carries the new name and
      ``proposed_new_collection_description`` carries a short sentence
      that will be stored alongside the collection row.

    Post-validation in :func:`classify_paper._resolve_raw` enforces the
    cross-field rules the strict-mode enum cannot (the chosen collection
    must be in range for the chosen domain; a new domain forces
    ``collection_index == -1``; ``proposed_new_domain_description`` must
    be non-empty iff ``domain_index == -1``;
    ``proposed_new_collection_description`` must be non-empty iff
    ``collection_index == -1``).
    """

    model_config = ConfigDict(extra="forbid")

    domain_index: int
    proposed_new_domain: str
    proposed_new_domain_description: str
    collection_index: int
    proposed_new_collection: str
    proposed_new_collection_description: str
    topics: list[str]


class ClassificationOutput(BaseModel):
    """Resolved classification, post index→name lookup.

    ``domain_description`` / ``collection_description`` hold the short
    descriptions the LLM supplied for newly-proposed taxonomy entries;
    each is ``None`` when the LLM picked an existing entry (no
    description to write) or when ``--domain-override`` forced a choice
    without LLM input (operator can fill it in later).
    """

    model_config = ConfigDict(extra="forbid")

    domain: str
    domain_is_new: bool
    domain_description: str | None
    collection: str
    collection_description: str | None
    topics: list[str]
