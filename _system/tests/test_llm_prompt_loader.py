"""Tests for :func:`_system.llm.prompt_loader.load_prompt`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from _system.llm import prompt_loader
from _system.llm.errors import PromptPlaceholderError
from _system.llm.prompt_loader import load_prompt


PROMPTS_DIR = Path(__file__).parent.parent / "llm" / "prompts"


def _write_prompt_set(
    root: Path,
    *,
    system: str = "sys",
    user: str = "usr",
    schema: dict | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "system.md").write_text(system, encoding="utf-8")
    (root / "user.md").write_text(user, encoding="utf-8")
    (root / "response.json").write_text(
        json.dumps(schema if schema is not None else {"name": "x", "description": "y", "schema": {}}),
        encoding="utf-8",
    )
    return root


@pytest.fixture
def prompts_root(tmp_path, monkeypatch):
    """Redirect load_prompt to look in a tmp_path-scoped prompts dir."""
    root = tmp_path / "prompts"
    root.mkdir()
    monkeypatch.setattr(prompt_loader, "PROMPTS_DIR", root)
    return root


def test_placeholder_substitution_in_markdown(prompts_root):
    _write_prompt_set(
        prompts_root / "my_case",
        system="hello {WHO}",
        user="topic = {TOPIC}; other = {WHO}",
    )
    loaded = load_prompt(
        "my_case", md_context={"WHO": "world", "TOPIC": "cats"}
    )
    assert loaded.system == "hello world"
    assert loaded.user == "topic = cats; other = world"


def test_schema_string_literal_replacement(prompts_root):
    schema = {
        "name": "n", "description": "d",
        "schema": {"enum_field": "REPLACE_ME"}
    }
    _write_prompt_set(prompts_root / "enum_case", schema=schema)
    loaded = load_prompt(
        "enum_case",
        schema_replacements={"REPLACE_ME": [1, 2, 3]},
    )
    assert loaded.schema["schema"]["enum_field"] == [1, 2, 3]


def test_unresolved_md_placeholder_raises(prompts_root):
    _write_prompt_set(prompts_root / "broken", user="value = {MISSING_KEY}")
    with pytest.raises(PromptPlaceholderError) as exc_info:
        load_prompt("broken", md_context={})
    msg = str(exc_info.value)
    assert "MISSING_KEY" in msg


def test_unresolved_schema_sentinel_raises(prompts_root):
    schema = {
        "name": "n", "description": "d",
        "schema": {"enum_field": "UNKNOWN_SENTINEL"}
    }
    _write_prompt_set(prompts_root / "schema_drift", schema=schema)
    with pytest.raises(PromptPlaceholderError) as exc_info:
        load_prompt("schema_drift", schema_replacements={})
    assert "UNKNOWN_SENTINEL" in str(exc_info.value)


def test_latex_bare_braces_in_body_do_not_trip_loader(prompts_root):
    latex_body = r"Consider $\{x : x > 0\}$ and use $y = {alpha}$ here."
    _write_prompt_set(
        prompts_root / "latex_case",
        user=f"body: {latex_body}\npaper: {{PAPER_CONTENT}}",
    )
    loaded = load_prompt(
        "latex_case", md_context={"PAPER_CONTENT": "body text"}
    )
    assert "body text" in loaded.user
    # LaTeX fragments survive verbatim.
    assert r"\{x : x > 0\}" in loaded.user
    assert "{alpha}" in loaded.user


def test_all_caps_token_in_substituted_content_does_not_raise(prompts_root):
    """Real papers contain LaTeX subscripts (`prompt_{IO}`, `p^{IO}`) and
    bibliographic tokens (`{ICML}`, `{ACL}`) that survive into the markdown.
    Once substituted into `{PAPER_CONTENT}`, those should not be re-flagged
    as unresolved placeholders — the leftover check must operate on the
    template, not the substituted result."""
    paper_with_caps_token = (
        "Standard input-output prompting formulates the task as "
        "$p_\\theta(y|\\texttt{prompt}_{IO}(x))$ where ..."
    )
    _write_prompt_set(
        prompts_root / "iocase",
        user="paper: {PAPER_CONTENT}",
    )
    loaded = load_prompt(
        "iocase", md_context={"PAPER_CONTENT": paper_with_caps_token}
    )
    assert "{IO}" in loaded.user
    assert "{PAPER_CONTENT}" not in loaded.user


def test_missing_prompt_dir_raises_file_not_found(prompts_root):
    with pytest.raises(FileNotFoundError):
        load_prompt("does_not_exist")


def test_missing_asset_raises_file_not_found(prompts_root):
    (prompts_root / "partial").mkdir()
    (prompts_root / "partial" / "system.md").write_text("s", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_prompt("partial")


# ---------------------------------------------------------------------------
# Regression tests on the real classify_paper asset (in-tree, not patched)
# ---------------------------------------------------------------------------


def test_real_classify_paper_assets_load_with_documented_context():
    md_context = {
        "EXISTING_TAXONOMY": (
            "0. rag — retrieval-augmented generation\n"
            "   └── 0: hierarchical indexing"
        ),
        "PAPER_CONTENT": "Abstract:\n... body ...\n\nIntroduction:\n...",
    }
    schema_replacements = {
        "DOMAIN_INDEX_ENUM": [-1, 0],
        "COLLECTION_INDEX_ENUM": [-1, 0],
    }
    loaded = load_prompt(
        "classify_paper",
        md_context=md_context,
        schema_replacements=schema_replacements,
    )
    assert "research librarian" in loaded.system
    assert "0. rag" in loaded.user
    assert "hierarchical indexing" in loaded.user
    assert loaded.schema["name"] == "classify_paper"
    props = loaded.schema["schema"]["properties"]
    assert props["domain_index"]["enum"] == [-1, 0]
    assert props["collection_index"]["enum"] == [-1, 0]


def test_real_classify_paper_user_asset_has_documented_placeholders():
    user_md = (PROMPTS_DIR / "classify_paper" / "user.md").read_text(encoding="utf-8")
    assert "{EXISTING_TAXONOMY}" in user_md
    assert "{PAPER_CONTENT}" in user_md


def test_real_classify_paper_response_json_has_documented_shape():
    schema_path = PROMPTS_DIR / "classify_paper" / "response.json"
    parsed = json.loads(schema_path.read_text(encoding="utf-8"))
    assert set(parsed.keys()) >= {"name", "description", "schema"}
    props = parsed["schema"]["properties"]
    assert set(props.keys()) == {
        "domain_index",
        "new_domain",
        "new_domain_desc",
        "collection_index",
        "new_collection",
        "new_collection_desc",
        "topics",
    }
    assert parsed["schema"]["additionalProperties"] is False
    assert parsed["schema"]["required"] == [
        "domain_index",
        "new_domain",
        "new_domain_desc",
        "collection_index",
        "new_collection",
        "new_collection_desc",
        "topics",
    ]
    assert props["domain_index"]["enum"] == "DOMAIN_INDEX_ENUM"
    assert props["collection_index"]["enum"] == "COLLECTION_INDEX_ENUM"
