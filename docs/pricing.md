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

## Where a cost comes from

`ProviderUsage.cost_source` is a `CostSource`. Claude turns are priced from the per-model counts in
the CLI's `modelUsage` and agentshim's table (`PRICE_TABLE`), because a CLI older than its model
bills the model at a fallback rate. `modelUsage` is cumulative over the process, so each turn is
priced by the difference, and its one-hour share of cache writes comes from the turn's
`usage.cache_creation`. A turn whose model has no table price has `total_cost_usd=None` and
`UNKNOWN_MODEL`. `PROVIDER_REPORTED` is the CLI's own figure, used only when its result named no
model. `None` means no cost was reported.

Known gaps: Claude's `result.usage` covers the main conversation only, so
tokens spent by a Task subagent are missing from `ProviderUsage.tokens` (they are
in the CLI's `modelUsage` and `total_cost_usd`). `ProviderUsage.by_model` has
them; see below. Copilot 1.0.83 prints no token counts.

## Usage by model

`ProviderUsage.by_model` is a tuple of `ModelUsage`, one per model that had
tokens in the turn, in the order the CLI listed them. It is a superset of
`tokens`: the rows' tokens add up to everything the turn spent, subagents and
advisors included, while `tokens` stays the main conversation (callers keep
feeding it to budgets, so it does not change).

| Field | Meaning |
| --- | --- |
| `model` | The id as the provider reports it (`claude-opus-5-5[1m]`, a dated snapshot). Normalize with `rates_for` or `price_for`. |
| `tokens` | This turn's tokens for the model. Only the primary row carries the turn's `turns` count, so the rows' turns add up to `tokens.turns`. |
| `recorded_usd` | The pinned-table price of `tokens` at the **standard tier**, never the CLI's own figure, and never a bill: it carries no speed premium whatever `speed` says. `None` when the table has no price for the model. Added in order, the rows are the turn's `total_cost_usd` when none is unpriced. |
| `speed` | The service speed (`UsageSpeed`), on the primary row only. `None` when the provider did not report one, or one agentshim has no name for; read it as standard. |
| `primary` | The model the turn ran on. Exactly one row of a turn is. |

- Claude: from `modelUsage`, priced exactly as the turn total is. One unpriced
  model still makes the turn `UNKNOWN_MODEL` with no total, but the other rows
  keep their prices. The primary is the model of the main conversation's latest
  model call, else the model in `system/init`, else the model with the most
  output. `result.usage.speed` is one value per turn, so it goes to the primary
  row. `thinkingTokens` fills `reasoning_output_tokens` when the CLI reports it.
  The one-hour share of cache writes is the turn's, applied to every model's
  writes, so a subagent's split is approximate. A steered message can start a
  follow-on turn inside one run: `modelUsage` is cumulative, so the rows are
  those of the last result, covering the whole run once, and `tokens` (which
  each result reports for itself) are the sum. A `modelUsage` key with no name
  becomes the row `unknown`, which has no price; the CLI's `<synthetic>`
  placeholder has no row.
- Codex app-server: one row at the model and service tier the server started the
  thread with. Child threads are not counted, as in `tokens`. The one-shot `exec`
  path knows no model and reports none.
- A provider that reports no split, or a turn whose increment is unknown, has
  `by_model == ()`.
- `merge_model_usage` adds the rows of two separate turns: rows with the same
  model and speed add up, and the first turn's primary stays the only primary.

### Billing a fast turn

`recorded_usd` and `total_cost_usd` are standard-tier list prices: the pinned
table has no speed tiers, and recorded cost does not move between releases. A
row with `speed` of `FAST` or `ULTRAFAST` was billed more than that, so a sum of
`recorded_usd` understates what a fast-tier user spent. To report the tier's
price, add the premium from the rate table (see
[Estimates at list prices](#estimates-at-list-prices)):

```python
entry = rates_for(read.table, "claude", row.model)
if entry is not None:
    tier = row.speed or UsageSpeed.STANDARD
    billed = price_tokens(row.tokens, entry.at(tier) or entry.standard).total_usd
    # or: row.recorded_usd + speed_premium_usd(row.tokens, entry, tier)
```

Use `price_tokens` at the tier for an estimate, and `recorded_usd` only for the
number agentshim recorded.

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

`agentshim.pricing` composes a snapshot of provider-supplied USD prices per million
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
  (`claude-opus-5-5[1m]`). Copilot and Gemini are not priced. Every model the
  catalogue lists has a price, or is named with its reason in the exclusion list of
  `tests/unit/test_pricing.py`.
- Reasoning is billed as output unless an entry sets `reasoning_usd`.
- A caller pins or extends assumptions with its own table:
  `default_pricing().with_entries([...], version="my-experiment")`, or
  `PricingTable.from_dict(json.load(...))`. Either way the table carries its
  own `version` and `last_updated`, so a report can say which prices it used.
- A plugin's `pricing_info(model, context, table)` method may compute a rate
  for a `PricingContext` (for example, a billing tier). Pass that context to
  `price_for(..., context=...)`. The returned `ModelPricing` records the source
  and checked date; a dynamic result need not appear in `default_pricing()`.
  The built-in static facts cover only the current global standard tier;
  historical `as_of` requests return `None` rather than pretending to know
  an earlier rate.

Every entry records `source` (the official pricing page's URL) and `checked`
(the date its numbers were last compared with that page); the table records
`version` and `last_updated`. All four are required: the constructors reject a
missing or malformed one, and `tests/unit/test_pricing.py` checks every
shipped entry. An entry that is not a published price sets `estimated=True`
and states its basis in `note`. The shipped table has no estimated entries.

Not modelled in the pinned table: OpenAI's long-context rates, Anthropic's
`inference_geo: "us"` 1.1x multiplier, fast mode and batch discounts. (Fast and
ultrafast rates are in `RateTable`, for estimates; recorded cost stays standard
tier, so it understates a fast turn. See [Billing a fast turn](#billing-a-fast-turn).)

### Updating the table

The Claude and Codex rows are generated, not edited. `scripts/generate_prices.py`
reads `tests/fixtures/litellm/snapshot.json`, a copy of
[LiteLLM's price file](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json)
trimmed to the models agentshim prices, with the day it was read and the SHA-256
of the whole file, and writes `providers/claude/prices_generated.py` and
`providers/codex/prices_generated.py`. A test regenerates them and fails on any
difference, so a hand edit cannot survive. The network is touched only by the
first step below, and a run never touches it.

1. Take a new snapshot and regenerate: `python scripts/generate_prices.py --fetch`
   (or `--from-file PATH` for a saved copy, with `--date`). It stops if LiteLLM
   has no row for a listed model. Add a new model by adding its LiteLLM key to
   `VENDORS` in the script; a name that is not listed is never priced, which is
   how `gemini-3-pro-preview` stays unpriced.
2. Read the diff of the generated files. Each changed number is a recorded cost
   that moves for callers who upgrade.
3. Run `python scripts/record_cost_golden.py` to accept those moves, and say so
   in the changelog: `tests/fixtures/recorded_costs.json` pins what every pinned
   rate, every spelling of a model name and the recorded Claude and Codex streams
   cost. Any difference you did not intend fails
   `tests/unit/test_recorded_cost_golden.py`.
4. The table's `version` is `litellm-<snapshot date>` and every entry's `checked`
   is that date, so a report can say which snapshot it priced with.
5. Notes and the rule that a missing cache write rate is priced at base input live
   in each provider's `model_data.py`. Where LiteLLM lists no write rate the
   entry says so in its `note`.
6. Run `uv run pytest tests/unit/test_pricing.py tests/unit/test_generate_prices.py`
   and note the change in `CHANGELOG.md`.

LiteLLM's rows for the pinned models match the vendors' pricing pages on every
value both have, which is why moving to a generated table changed no price. The
older OpenAI rows have no cache write rate in LiteLLM and keep the input rate.

## Estimates at list prices

Recorded cost is what a turn cost when it ran, at the pinned table, and it never
changes afterwards. An *estimate* is what the same tokens would cost at today's
list prices, for a report. The two are different types on purpose, so a fetched
rate cannot reach a recorded cost:

| | `PricingTable` | `RateTable` |
| --- | --- | --- |
| Purpose | Recorded cost | Estimate |
| Key | `(vendor, model)` | One flat, lowercase name |
| Rows | The models agentshim prices, standard tier | Everything in LiteLLM's file, plus bare aliases, with fast and ultrafast tiers |
| Zero input rate | Refused | Allowed (a free model), but never over a pinned price |
| Source | Generated from a snapshot at release | A fetched document laid over the pinned table |
| Used by | `price_for`, `cost_usd` | `rates_for`, `price_tokens` |

`price_for` and `cost_usd` take a `ModelPricing` only, and `price_tokens` takes
`Rates` only.

```python
from agentshim import (
    RateStatus, TokenUsage, price_tokens, rates_for, read_rates, refresh_rates,
)

read = read_rates(store, now=now)            # never touches the network
if read.refresh_due:
    read = refresh_rates(store, source, now=now)
entry = rates_for(read.table, "claude", "claude-opus-5-5[1m]")
if entry is not None:
    cost = price_tokens(tokens, entry.standard)      # CostBreakdown by token class
```

- `parse_litellm(document, checked_on=...)` reads LiteLLM's file. A model needs
  finite, non-negative input and output rates or it is dropped; a speed tier
  with no input or output price under a priced standard is dropped too. A missing cache
  read or write rate defaults to the input rate. Fast comes from
  `provider_specific_entry.fast` (a multiple of standard) or the `_priority`
  columns, ultrafast from `_ultrafast`, and a tier's missing cache rates scale by
  the standard ratio. One-hour cache writes come from
  `cache_creation_input_token_cost_above_1hr`. Long-context, batch and flex
  columns are not read, so estimates for requests past a model's long-context
  line are low. Bare names (`gpt-5.4` for `openai/gpt-5.4`) are added as aliases
  when the sources agree; where two reseller rows differ the alias is dropped.
  The parser raises only `RateDocumentError`, whatever the input.
- agentshim holds no HTTP client and picks no directory. The caller passes a
  `RateSource` (`fetch(endpoint) -> bytes`, raising `RateFetchError`) and a
  `RateStore` (`read()` and `write(snapshot)`). `FakeRateSource`,
  `FakeRateStore` and `RateStoreContract` are in `agentshim.testing`.
  `LITELLM_ENDPOINTS` lists the raw file, then GitHub's API for the same file;
  the second needs the `Accept` header its `RateEndpoint` carries.
- A fetch is due when the stored document is older than `RATES_TTL` (24 hours),
  or `force` is set, and no attempt, failed or not, was made within
  `REFRESH_FLOOR` (60 seconds). A document that does not parse, or parses to no
  model, counts as a failed fetch and never replaces a good one.
  `RateStatus` says how current the document behind `read.table` is: `FRESH`,
  `CACHED` (older, or the latest refresh failed) or `BUNDLED` (no usable
  document: the pinned table shipped in the wheel, with its speed tiers).
- A fetched document is laid over the pinned table, so it never takes a known
  price away. A model the document lacks keeps its pinned rates (LiteLLM renames
  and drops rows), and a row with no input or output price does not replace a
  pinned price (LiteLLM lists placeholders, and rows for models that are not
  chat models). Every model the pinned table prices is priced in `read.table`;
  a model only the document knows comes as the document lists it, free or not.
- `now` must be timezone-aware (`ValueError` otherwise, before anything is
  fetched). A clock set back is tolerated: a time stamped ahead of `now` by more
  than the floor has an unknown age and counts as expired, and an attempt within
  the floor of `now` on either side holds the floor.
- `rates_for(table, provider, model)` maps the provider's names onto the table's
  flat names: a bracketed variant is dropped, a dated snapshot falls back to its
  undated name, and a `vendor/` prefix is tried last. A name that is not a model
  (Claude's `<synthetic>`, the bare family names `opus`, `sonnet`, `haiku`,
  `fable`) has no rates. A plugin customizes this with `rate_names`; see
  [Extending](extending.md).
- `price_tokens(tokens, rates)` bills every token class once and puts reasoning
  at the output rate. `speed_premium_usd` and `cache_savings_usd` give the other
  two figures a report shows.

### Why agentshim holds prices

Pricing was first meant to stay out of agentshim, with callers supplying
their own weights. At the user's direction (2026-09-27), agentshim instead
maintains this static, versioned table next to the counts it normalizes, so
every caller prices the same counts the same way and states which table it
used. Callers still override it for experiments.
