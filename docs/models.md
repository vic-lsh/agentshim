# Models

agentshim holds two kinds of model facts, so a caller never hard-codes them.

## Which models a provider runs

`ProviderProfile.models` lists the model ids a provider's CLI takes in its
`--model` flag and that agentshim has run, the provider's default first.
`ProviderProfile.default_model` is that first entry, or `None` when the list
is empty. An empty list means agentshim knows of no model for that CLI; a
caller may still pass one, at its own risk.

```python
from agentshim import get_provider

profile = get_provider("claude").profile
profile.models         # ("claude-opus-5", "claude-opus-5-5", ...)
profile.default_model  # "claude-opus-5"
```

## Context windows

`agentshim.core.models` ships a static, versioned table of context windows in
tokens, one `ModelSpec` per `(vendor, model)`, each with the official page it
was read from (`source`) and the date it was checked (`checked`).

```python
from agentshim import context_window, model_spec

context_window("claude", "claude-opus-5-5")   # 1_000_000
context_window("codex", "gpt-6-luna")         # 1_050_000
context_window("claude", "claude-opus-9")     # None: not known
```

- The provider resolves to a vendor as pricing does (`codex` runs OpenAI
  models, `claude` Anthropic ones); a `vendor/model` name (opencode) names the
  vendor itself (`google/` and `google-vertex/` are Google), and a trailing date snapshot (`-20251001`) is dropped.
- An unknown model returns `None`. The lookup never matches by prefix or
  family and never falls back to another model's window. A bracketed variant
  (`claude-opus-5-5[1m]`) returns `None`, because a variant may change the
  window.
- A caller pins or extends the table with
  `default_models().with_entries([...], version="mine")` and passes it as the
  `table` argument.

## Display names

`model_display_name(provider, model)` returns how a person reads the model
(`"Opus 5.5"`, `"GPT-6.1 Sol"`), from the same entry and with the same exact
lookup as `context_window`; an unknown model returns `None`, and the caller
decides what to show. Every shipped entry has a `display_name`.

### Updating the table

1. Read each model's window from its vendor's page:
   - OpenAI: `https://developers.openai.com/api/docs/models/<model>`
     ("context window");
   - Anthropic: <https://platform.claude.com/docs/en/build-with-claude/context-windows>
     ("Context window sizes by model");
   - Google: `https://ai.google.dev/gemini-api/docs/models/<model>`
     ("Input token limit").
2. Give each entry its vendor's name for the model as `display_name`. Edit `DEFAULT_MODELS` in `agentshim/core/models.py` and set `_CHECKED`, the
   table's `version` and `last_updated` to today's date.
3. Where a window is not published, leave the model out rather than guess.
4. A model added to a provider's `ProviderProfile.models` needs a window;
   `tests/unit/test_models.py` fails otherwise.
