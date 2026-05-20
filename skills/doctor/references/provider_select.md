# Provider selection (read me after `check-config.sh`)

You're here because `~/.config/lodestone/config.toml` is missing or
incomplete and `check-config.sh` returned a JSON object. Inspect the
`providers_with_keys` array and follow the matching branch.

## Zero providers — `providers_with_keys: []`

Tell the user, then stop:

> No provider API key is set. Lodestone's classify stage needs exactly
> one of `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY`
> exported in this shell. Set one and re-run `/lodestone:doctor`.

Do not call any further tools. Do not write `config.toml`.

## One provider — `providers_with_keys: ["<p>"]`

Skip provider selection. Set `provider = providers_with_keys[0]` and go
straight to `references/model_select.md`.

If the check-config JSON also includes a non-null `config_provider`
(meaning the config already pins a provider but no model), use that
value as `provider` instead — don't ask the user to re-pick.

## Multiple providers — `len(providers_with_keys) > 1`

Call `AskUserQuestion` exactly once:

- `question`: `"Lodestone runs one LLM call per ingestion: which provider should I use?"`
- `header`: `"Provider"`
- `multiSelect`: `false`
- `options`: one entry per provider name in `providers_with_keys`, in
  the order returned. Use the provider name as `label` and a one-line
  description noting which env var supplies the key:
  - `anthropic` → `"uses $ANTHROPIC_API_KEY"`
  - `openai` → `"uses $OPENAI_API_KEY"`
  - `gemini` → `"uses $GEMINI_API_KEY"`

The harness always appends an "Other" option — you cannot suppress it.
**Guardrail:** if the user picks Other and types a provider name that
isn't in `providers_with_keys`, refuse and re-ask. There is no API key
for that provider in the environment, so we cannot proceed with it.
Re-ask by calling `AskUserQuestion` again with the same options.

Once the user picks a valid provider, set `provider` to their choice
and proceed to `references/model_select.md`.
