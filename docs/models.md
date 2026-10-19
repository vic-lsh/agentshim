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

`agentshim.models` composes a versioned snapshot of provider-supplied context windows in
tokens, one `ModelSpec` per `(vendor, model)`, each with the official page it
was read from (`source`) and the date it was checked (`checked`).

```python
from agentshim import context_window, model_spec

context_window("claude", "claude-opus-5-5")   # 1_000_000
context_window("codex", "gpt-6-luna")         # 1_050_000
context_window("claude", "claude-opus-9")     # None: not known
```

- The provider resolves to a vendor through its catalog plugin (`codex` runs OpenAI
  models, `claude` Anthropic ones); a `vendor/model` name (opencode) names the
  vendor itself (`google/` and `google-vertex/` are Google), and a trailing date snapshot (`-20251001`) is dropped.
- An unknown model returns `None`. The lookup never matches by prefix or
  family and never falls back to another model's window. A bracketed variant
  (`claude-opus-5-5[1m]`) returns `None`, because a variant may change the
  window.
- A caller pins or extends the table with
  `default_models().with_entries([...], version="mine")` and passes it as the
  `table` argument.
- A plugin's `model_info(model, table)` method may calculate a model's facts
  even when that model is not in the enumerated snapshot. It must still return
  a `ModelSpec` with a source and checked date.

## Names, families and tiers

Each shipped `ModelSpec` also names the model for a picker: `display_name`
(`"Opus 5.5"`), `family` (`"Claude"`, `"GPT"`, `"Gemini"`), `tier` and
`tier_name`. A tier is a rank within a family, `1` the smallest and cheapest
(Haiku < Sonnet < Opus < Fable; Luna < Terra < Sol < Astra), shared by every
version of that tier. A model that sits on no ladder (`gpt-5.5`) has no tier.
A harness's readable name is `ProviderProfile.display_name`.

```python
spec = model_spec("claude", "claude-opus-5-5")
spec.display_name, spec.family, spec.tier, spec.tier_name
# ("Opus 5.5", "Claude", 3, "Opus")
```

A caller-built entry may leave all four `None`; `display_name` and `family`
are set together, as are `tier` and `tier_name`, and a tier needs a family.

## Generations

Each shipped `ModelSpec` also places the model in a generation of its vendor:
`generation_id` (`"5.5"`), `generation` (`"Claude 5.5"`, `"GPT-6"`) and
`generation_rank`, `1` for the vendor's newest. Every model of one generation
shares all three (`claude-opus-5-5` and `claude-sonnet-5-5` are both Claude
5.5), and one vendor's generations have distinct ranks, which `ModelTable`
checks. The three are set together and may all be `None` on a caller-built
entry. A picker groups by `generation_id` and orders groups by
`generation_rank`, then orders models within a group by `tier`.

```python
spec = model_spec("codex", "gpt-6.1-sol")
spec.generation_id, spec.generation, spec.generation_rank
# ("6.1", "GPT-6.1", 1)
```

### Updating the table

1. Read each model's window from its vendor's page:
   - OpenAI: `https://developers.openai.com/api/docs/models/<model>`
     ("context window");
   - Anthropic: <https://platform.claude.com/docs/en/build-with-claude/context-windows>
     ("Context window sizes by model");
   - Google: `https://ai.google.dev/gemini-api/docs/models/<model>`
     ("Input token limit").
2. Edit `MODEL_ENTRIES` in the relevant provider's `model_data.py` and update
   `_MODEL_CHECKED`. Update the composed table version in `agentshim/models.py`
   when facts change.
3. Give each new model its `display_name`, `family`, tier and generation (see above). A
   new generation takes rank 1 and pushes the older ones down by one.
4. Where a window is not published, leave the model out rather than guess.
5. A model added to a provider's `ProviderProfile.models` needs a window;
   `tests/unit/test_models.py` fails otherwise.
