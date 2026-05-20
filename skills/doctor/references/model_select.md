# Model selection + config write (read me after `provider_select.md`)

You've resolved a single `provider` (one of `anthropic` / `openai` /
`gemini`). Now elicit a model and persist the choice to
`~/.config/lodestone/config.toml`.

## Model catalogs

### Preferred path — web-search for current model IDs

Provider model IDs change constantly (Anthropic, OpenAI, and Google all
ship monthly). The catalog below this section is a frozen snapshot and
will drift. **Always try WebSearch first** and only fall back to the
frozen catalog when WebSearch is unavailable or its results can't be
parsed into a confident set of current IDs.

Call `WebSearch` with a query targeted at the chosen provider's
*official* model-listing documentation. One query is enough:

- **anthropic**: `Anthropic Claude API model names site:docs.anthropic.com`
- **openai**: `OpenAI API model IDs site:platform.openai.com/docs/models`
- **gemini**: `Google Gemini API model versions site:ai.google.dev`

From the results, extract 3–4 production-ready model IDs **in the
provider's recommended order** (the docs invariably lead with the
flagship). Each row needs:
- the literal model ID string (e.g. `claude-opus-4-7`, *not* a friendly
  name like "Claude Opus 4.7")
- a one-clause description (most capable / balanced / fastest / cheapest)

Quality bar before using web results:
- IDs must come from the provider's own docs domain (anthropic.com,
  openai.com, ai.google.dev) — not blog roundups or third-party listings.
- If the search returns fewer than 2 confident IDs, or the IDs look
  like marketing names rather than API strings, fall back.
- Preview/experimental IDs are acceptable as the 4th slot but never as
  the recommended/first slot.

Tag the first row (the flagship) with `(Recommended)` in the
AskUserQuestion option label.

### Fallback catalog (used only if WebSearch fails)

Recommended is listed first. This snapshot drifts over time — only use
when the preferred path above didn't produce a usable set.

- **anthropic**
  - `claude-opus-4-7` — most capable (Recommended)
  - `claude-sonnet-4-6` — balanced speed + intelligence
  - `claude-haiku-4-5` — fastest, lowest cost
- **openai**
  - `gpt-5.4` — most capable (Recommended)
  - `gpt-5.4-mini` — faster, lower cost
  - `gpt-5.4-pro` — deepest reasoning
  - `gpt-5.4-nano` — cheapest, fastest
- **gemini**
  - `gemini-2.5-pro` — most capable GA (Recommended)
  - `gemini-2.5-flash` — fast, balanced
  - `gemini-2.5-flash-lite` — fastest, lowest cost
  - `gemini-3.1-pro-preview` — preview, experimental frontier

## Ask the user

Call `AskUserQuestion` exactly once:

- `question`: `"Lodestone makes one LLM call per ingestion, which {provider} model should classify-stage use?"`
  (substitute the provider name)
- `header`: `"Model"`
- `multiSelect`: `false`
- `options`: the rows you assembled via WebSearch (preferred) or the
  fallback catalog, in the order described above. Label the flagship
  row with the `(Recommended)` suffix per the AskUserQuestion
  convention. Cap at 4 options — the harness limit.

The harness automatically appends an "Other" option. **Allow it here**
— that slot lets the user type a custom model ID (e.g. a snapshot or
preview not yet in the catalog). Trust the string they enter verbatim.

## Persist `config.toml`

After the user picks (catalog or Other), write
`~/.config/lodestone/config.toml`. Resolve the path the same way
lodestone does:

1. If `$XDG_CONFIG_HOME` is set, use `$XDG_CONFIG_HOME/lodestone/config.toml`.
2. Otherwise on POSIX, use `~/.config/lodestone/config.toml`.
3. On Windows, use `%APPDATA%/lodestone/config.toml` (fall back to
   `~/.config/lodestone/config.toml` if `APPDATA` is unset).

Create the parent directory if it doesn't exist. Body:

```toml
[llm]
provider = "<provider>"
model = "<model>"
temperature = 1.0
```

Escape any backslashes or double-quotes in `<model>` for TOML basic
strings (`\` → `\\`, `"` → `\"`). Catalog IDs don't need escaping;
Other-supplied IDs might.

After the write, return to the calling SKILL.md step.
