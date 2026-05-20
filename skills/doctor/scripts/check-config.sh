#!/usr/bin/env bash
# LLM provider config probe for /lodestone:doctor.
#
# Adapted from deep-sota's scripts/checks/validate-env.sh. Emits the
# same JSON shape so doctor and the duplicated references files
# (provider_select.md / model_select.md) can be reused unchanged.
#
# Exit codes (forwarded from check-config.py):
#   0 — valid
#   2 — config file malformed
#   3 — config pins a provider whose API key is not set
#   4 — config missing AND zero provider keys in env
#   5 — config missing OR incomplete; ≥1 provider keys present
#   6 — uv not installed (set here, before invoking Python)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if ! command -v uv >/dev/null 2>&1; then
    printf '{"valid": false, "errors": ["uv not installed. Install from https://docs.astral.sh/uv/"], "warnings": [], "config_path": null, "config_status": "missing", "config_provider": null, "config_model": null, "providers_with_keys": []}\n'
    exit 6
fi

# --python 3.11 matches lodestone's floor (tomllib requires 3.11+).
# --no-project keeps this a stdlib-only invocation so it works before
# the plugin's own venv exists.
exec uv run --python 3.11 --no-project "$SCRIPT_DIR/check-config.py"
