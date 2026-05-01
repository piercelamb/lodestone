"""Strict prompt asset loader.

Each use case owns a directory under ``_system/llm/prompts/<name>/``
containing:

- ``system.md``   — system prompt; supports ``{PLACEHOLDER}`` substitution
- ``user.md``     — user prompt; same substitution rules
- ``response.json`` — OpenAI-style ``{"name", "description", "schema"}``
  JSON. String *values* that equal a key in ``schema_replacements`` are
  replaced with that key's value (which may itself be any JSON-serializable
  type — list, int, dict).

``str.replace`` is used for the .md substitution, not ``str.format`` —
real paper content contains bare ``{}`` sequences from LaTeX that would
choke ``format``.

After substitution, the loader scans both the markdown and the JSON for
unresolved placeholders. Anything still matching ``{ALL_CAPS}`` in the
markdown or still-matching-a-key in the JSON raises
:class:`PromptPlaceholderError` naming the key and path — silent prompt
defects cost money and time.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, NamedTuple

from _system.llm.errors import PromptPlaceholderError


PROMPTS_DIR = Path(__file__).parent / "prompts"

# A leftover placeholder in a .md file after substitution. ALL CAPS with
# optional underscores/digits inside the braces — narrow enough that
# LaTeX fragments like `{x}` or `{\alpha}` don't false-positive.
_UNSUB_MD_PLACEHOLDER_RE = re.compile(r"\{[A-Z][A-Z0-9_]*\}")


class LoadedPrompt(NamedTuple):
    system: str
    user: str
    schema: dict[str, Any]  # {"name", "description", "schema"}


def _substitute_markdown(
    text: str, md_context: dict[str, str], *, path: Path
) -> str:
    # Validate against the template, not the substituted output. After
    # substitution, an `{ALL_CAPS}` token in the result could be either an
    # unfilled slot OR user-supplied content (LaTeX subscripts like `{IO}`,
    # bibliographic `{ICML}` tokens, etc.) — and we can't tell them apart
    # post-hoc. Checking pre-substitution makes the rule unambiguous: every
    # placeholder the template declares must be backed by a context entry.
    declared = set(_UNSUB_MD_PLACEHOLDER_RE.findall(text))
    provided = {"{" + key + "}" for key in md_context}
    missing = sorted(declared - provided)
    if missing:
        raise PromptPlaceholderError(
            f"unresolved placeholder {missing[0]!r} in {path}"
        )
    out = text
    for key, value in md_context.items():
        out = out.replace("{" + key + "}", str(value))
    return out


def _substitute_schema(
    node: Any, replacements: dict[str, object]
) -> Any:
    """Walk the JSON tree, replacing string literals that match a key."""
    if isinstance(node, dict):
        return {k: _substitute_schema(v, replacements) for k, v in node.items()}
    if isinstance(node, list):
        return [_substitute_schema(v, replacements) for v in node]
    if isinstance(node, str) and node in replacements:
        return replacements[node]
    return node


def _find_unresolved_schema_placeholders(
    node: Any, replacement_keys: set[str]
) -> str | None:
    """Return the first unresolved placeholder literal found in the tree.

    Exhaustive walk is already done by ``_substitute_schema``, so any string
    that still equals a declared ``schema_replacements`` key means the walk
    was bypassed (can't happen in practice). The more useful catch is
    author-side typos: a ``DOMAIN_INDEX_ENUM``-style sentinel left in the
    JSON that nobody declared. Heuristic: ``SNAKE_CASE_UPPER`` tokens (at
    least one underscore) are almost certainly placeholders, while common
    uppercase enum literals like ``"ACTIVE"`` are not.
    """
    sentinel_re = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")

    stack: list[Any] = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
        elif isinstance(cur, str):
            if cur in replacement_keys:
                return cur
            if sentinel_re.match(cur):
                return cur
    return None


def load_prompt(
    name: str,
    *,
    md_context: dict[str, str] | None = None,
    schema_replacements: dict[str, object] | None = None,
) -> LoadedPrompt:
    """Load ``system.md`` / ``user.md`` / ``response.json`` for *name*."""
    md_context = md_context or {}
    schema_replacements = schema_replacements or {}

    prompt_dir = PROMPTS_DIR / name
    if not prompt_dir.is_dir():
        raise FileNotFoundError(
            f"prompt directory not found: {prompt_dir}"
        )

    system_path = prompt_dir / "system.md"
    user_path = prompt_dir / "user.md"
    schema_path = prompt_dir / "response.json"
    for p in (system_path, user_path, schema_path):
        if not p.is_file():
            raise FileNotFoundError(f"missing prompt asset: {p}")

    system = _substitute_markdown(
        system_path.read_text(encoding="utf-8"),
        md_context,
        path=system_path,
    )
    user = _substitute_markdown(
        user_path.read_text(encoding="utf-8"),
        md_context,
        path=user_path,
    )

    schema_raw = json.loads(schema_path.read_text(encoding="utf-8"))
    schema = _substitute_schema(schema_raw, schema_replacements)

    leftover = _find_unresolved_schema_placeholders(
        schema, set(schema_replacements)
    )
    if leftover is not None:
        raise PromptPlaceholderError(
            f"unresolved schema placeholder {leftover!r} in {schema_path}"
        )

    return LoadedPrompt(system=system, user=user, schema=schema)
