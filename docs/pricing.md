# Token Usage and Pricing

## Normalized counts

Every provider's counts are folded into one `TokenUsage` breakdown:

| Field | Meaning |
| --- | --- |
| `input_tokens` | Every input token, including cache reads and cache writes. |
| `cache_read_input_tokens` | Input served from the prompt cache. |
| `cache_write_input_tokens` | Input written to the prompt cache this turn. |
| `cache_write_1h_input_tokens` | The part of the writes with a one-hour TTL (Anthropic's extended cache). |
| `uncached_input_tokens` | Derived: `input - cache_read - cache_write`, billed at the base rate. |
| `output_tokens` | Every generated token, reasoning included. |
| `reasoning_output_tokens` | The part of the output the provider reports as reasoning. |
| `turns` | Model turns, as the provider counts them. |
| `cached_input_tokens` | Deprecated alias of `cache_read_input_tokens`. |

`TokenUsage` checks on construction that
`cache_read + cache_write <= input`, `cache_write_1h <= cache_write` and
`reasoning <= output`. Parsers build it with `normalized_usage`, which clamps a
CLI's inconsistent counts instead of failing the turn.

How each CLI's report maps onto it:

| Provider | Input | Cache reads | Cache writes | Output / reasoning |
| --- | --- | --- | --- | --- |
| Codex (`turn.completed`) | `input_tokens` already includes reads and writes | `cached_input_tokens` | `cache_write_input_tokens` | `output_tokens` already includes `reasoning_output_tokens` |
| Claude (`result.usage`) | `input_tokens` + reads + writes (reported disjoint) | `cache_read_input_tokens` | `cache_creation_input_tokens`; 1h part from `cache_creation.ephemeral_1h_input_tokens` | `output_tokens` includes thinking; reasoning is not reported apart (0) |
| opencode (`step_finish`) | `input` + `cache.read` + `cache.write` | `cache.read` | `cache.write` | `output` + `reasoning` |
| Copilot (`assistant.usage`) | `inputTokens` + reads + writes | `cacheReadTokens` | `cacheWriteTokens` | `outputTokens` + `reasoningTokens` |
| Gemini (`result.stats`) | `input_tokens` already includes `cached` | `cached` | not reported (0) | `output_tokens`; thinking is not reported |

Codex's five fields are cumulative thread totals, including on resume.
The parser subtracts the previous raw report to produce per-invocation counts;
`ProviderUsage.raw` retains the cumulative report. A resume without a baseline
returns normally with `increment_known=False` and zero token placeholders
(completion frames still count in `turns`). Check this marker before pricing
or applying token budgets; the unknown invocation is not known to be free.
The retained raw total supplies the next invocation's baseline. The marker is
included in `ProviderUsage.to_dict()`. See
[provider behavior](providers.md#token-usage) for checkpointed resumes.

Known gaps: Claude's `result.usage` covers the main conversation only, so
tokens spent by a Task subagent are missing from it (they are in the CLI's
`modelUsage` and `total_cost_usd`). Copilot 1.0.83 prints no token counts.

`to_dict()` only ever adds keys. `TokenUsage.from_dict` also reads a mapping
written before 0.7.0, whose `cached_input_tokens` counted cache writes too on
Claude, opencode and Copilot.

## Weighted totals

`TokenUsage.weighted_total(TokenWeights(...))` multiplies each token class by
a caller-supplied weight. The classes partition the turn (uncached input,
cache reads, five-minute and one-hour cache writes, non-reasoning output and
reasoning output), so no token is counted twice. Weights have no unit: USD per
million tokens and multiples of the base input rate both work.

## Pricing table

`agentshim.core.pricing` ships a static table of USD prices per million
tokens, one `ModelPricing` per `(vendor, model)`, at the vendor's standard
(non-batch, global, short-context) tier.

```python
from agentshim import cost_usd, default_pricing, price_for

pricing = price_for("codex", "gpt-6-luna")  # None when the model is unknown
if pricing is not None:
    print(cost_usd(result.usage.tokens, pricing))
    print(pricing.relative_weights())          # multiples of base input
table = default_pricing()
print(table.version, table.last_updated)
```

- An unknown model returns `None`; the table never falls back to another
  model's price or to zero.
- `price_for` accepts a provider (`codex` bills through OpenAI, `claude`
  through Anthropic) or a vendor name, a `vendor/model` name (opencode), a
  dated snapshot (`claude-haiku-4-5-20251001`) and a bracketed variant
  (`claude-opus-5-5[1m]`). Copilot and Gemini are not priced.
- Reasoning is billed as output unless an entry sets `reasoning_usd`.
- A caller pins or extends assumptions with its own table:
  `default_pricing().with_entries([...], version="my-experiment")`, or
  `PricingTable.from_dict(json.load(...))`. Either way the table carries its
  own `version` and `last_updated`, so a report can say which prices it used.

Every entry records `source` (the official pricing page's URL) and `checked`
(the date its numbers were last compared with that page); the table records
`version` and `last_updated`. All four are required: the constructors reject a
missing or malformed one, and `tests/unit/test_pricing.py` checks every
shipped entry. An entry that is not a published price sets `estimated=True`
and states its basis in `note`. The shipped table has no estimated entries.

Not modelled: OpenAI's long-context rates, Anthropic's `inference_geo: "us"`
1.1x multiplier, fast mode and batch discounts.

### Updating the table

1. Open each vendor's official page:
   - OpenAI: <https://developers.openai.com/api/docs/pricing> (standard tier,
     short-context column);
   - Anthropic: <https://platform.claude.com/docs/en/about-claude/pricing>
     (Model pricing table: base input, 5m cache writes, 1h cache writes,
     cache hits, output).
2. Edit `DEFAULT_PRICING` in `agentshim/core/pricing.py`: change the prices
   that moved, add new models, and set `_CHECKED` (or an entry's own
   `checked`) to today's date for every entry you compared. Keep `source`
   pointing at the page you read.
3. Set the table's `version` and `last_updated` to today's date. Change
   `version` whenever any price changes, so reports stay reproducible.
4. Where a published price is missing, leave the model out rather than guess.
   If a caller needs one anyway, add it with `estimated=True` and the basis in
   `note`.
5. Run `uv run pytest tests/unit/test_pricing.py` and note the change in
   `CHANGELOG.md`.

### Why agentshim holds prices

Pricing was first meant to stay out of agentshim, with callers supplying
their own weights. At the user's direction (2026-09-27), agentshim instead
maintains this static, versioned table next to the counts it normalizes, so
every caller prices the same counts the same way and states which table it
used. Callers still override it for experiments.
