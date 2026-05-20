#!/usr/bin/env bash
# HuggingFace model cache check for /lodestone:doctor.
#
# Calls validate_models --check-only inside the lodestone venv, which
# uses snapshot_download(local_files_only=True) to verify completeness
# (catches partial downloads, missing variant files, and stale
# revisions that bare directory-existence checks would false-positive).
#
# Emits one line per model — "bge cached" / "bge NOT cached" /
# "gliner cached" / "gliner NOT cached" — for SKILL.md to interpret.
#
# Falls back to a bare directory-existence check when the lodestone
# venv is missing (huggingface_hub can't be imported without it).
# That case is already surfaced by doctor's Plugin venv check above,
# so the fallback's only job here is to keep the diagnostic readable.
#
# Always exits 0 — the JSON-shaped status lives in the output, and
# SKILL.md's `!` injection treats non-zero exit as "shell command
# failed" instead of parsing the output.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLUGIN_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [[ ! -d "$PLUGIN_ROOT/.venv" ]]; then
    # Loose fallback — venv hasn't been prewarmed yet, so we can't run
    # the rich check. Plugin venv check above this in SKILL.md will be
    # FAILed too; its remediation populates the venv, after which a
    # re-run of doctor uses the rich path.
    test -d "$HOME/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5" \
        && echo "bge cached (loose check — venv not ready, completeness not verified)" \
        || echo "bge NOT cached"
    test -d "$HOME/.cache/huggingface/hub/models--fastino--gliner2-large-v1" \
        && echo "gliner cached (loose check — venv not ready, completeness not verified)" \
        || echo "gliner NOT cached"
    exit 0
fi

uv run --no-sync --directory "$PLUGIN_ROOT" python -m _system.scripts.validate_models --check-only 2>/dev/null || {
    # Rich check blew up (huggingface_hub import error, broken venv,
    # etc.). Fall back to the loose check so doctor still surfaces
    # something actionable.
    test -d "$HOME/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5" \
        && echo "bge cached (loose check — rich verify errored)" \
        || echo "bge NOT cached"
    test -d "$HOME/.cache/huggingface/hub/models--fastino--gliner2-large-v1" \
        && echo "gliner cached (loose check — rich verify errored)" \
        || echo "gliner NOT cached"
}
exit 0
