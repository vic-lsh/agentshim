# Account Limits

A subscription login has limits that are not a model's: a session window of a few
hours, a weekly one, sometimes one per model, each with a share used and a time it
refills. `read_limits` asks the provider's CLI what they are right now. It starts the
CLI with no prompt, asks one question and stops it; no model runs and nothing is spent.

```python
from agentshim import read_limits

limits = read_limits("claude")            # or "codex"; env defaults to the login shell's
if limits.unavailable is not None:
    print(limits.unavailable.reason, limits.unavailable.detail)
for window in limits.windows:
    print(window.label, f"{window.used_fraction:.0%}", window.resets_at)
```

A read never raises for a CLI problem. It returns `unavailable` and no windows, so a
caller keeps its last good reading and says why it is old:

| Reason | Meaning |
| --- | --- |
| `unsupported` | The provider has no limits read (OpenCode, Gemini, Copilot). |
| `api_key` | The login is billed by token, so no plan applies. |
| `probe_failed` | No CLI, not logged in, timed out, or an error answer; `detail` says which. |

`read_limits` takes the arguments `probe_provider` takes (`executor`, `confinement`,
`env`), so a read runs where a turn would run, plus `timeout` for the whole read and a
`clock`. An unknown provider is a `ValueError`.

## What a window is

| Field | Meaning |
| --- | --- |
| `id` | Stable for the same window across reads and across a turn's events. |
| `kind` | `session`, `weekly`, `monthly` or `other`, from the window's length. |
| `label` | `Session`, `Weekly`, `Monthly`, or `Weekly · <model>` for a per-model window. |
| `used_fraction` | Share used, `0.0` to `1.0` (above 1 when the provider reports over-use). |
| `resets_at` | When it refills, timezone aware, or `None` when not stated. |
| `window_minutes` | The window's length when stated or fixed by its name. |

`AccountLimits` also carries `plan` (the provider's own word, `team`, `pro`),
`usage_url` (where a person sees the same numbers) and `account`, a short digest of the
account (a hash of the email for Claude, of the account id for Codex). Pool windows of
one account seen from several machines by `account`; neither the email nor the id is
returned. A window the provider did not state a share for is left out, never zero.

## Merging with readings from turns

A turn emits a `RateLimitStatus` for a window as it runs. `limit_window(provider, status)`
returns the `LimitWindow` a read would have returned for it, with the same `id`, so a
caller that keeps both merges by `id` and never learns `five_hour` or `primary`.
A status with no window, or no share used and no sign the limit is reached, is `None`.

## Reset credits

Some Codex plans hold reset credits, each of which refills the limits at once.
`AccountLimits.reset_credits` has `available`, the count the provider states, and
`credits`, the ones it described (a list that can be shorter than the count).

`use_reset_credit(provider, idempotency_key=..., credit_id=None)` spends one and reads the
limits again. It is a separate function, and a read has no path to it, because it
costs the account something that cannot be undone.

- `idempotency_key` names one logical attempt. A caller that retries after an error it
  cannot read reuses the key, and the provider spends at most one credit for it. Make a
  new key (a UUID) for each separate decision to spend.
- `credit_id` names the credit (`ResetCredit.id`); `None` lets the provider pick.
- The result is `ResetCreditResult(outcome, limits)`. `outcome` is `reset`,
  `nothing_to_reset`, `no_credit` or `already_redeemed`. An outcome agentshim does not
  know raises `TurnFailedError`: the credit may or may not be spent, so read the limits.
- A provider without credits (Claude) raises `ProviderCapabilityError`.

## What each provider does

| Provider | Mechanism | Needs |
| --- | --- | --- |
| `claude` | A `get_usage` control request over `stream-json`, with `skip_behaviors`, after `initialize`. | CLI 2.1.304 or later. The SDK calls it `usage_EXPERIMENTAL_MAY_CHANGE_DO_NOT_RELY_ON_THIS_API_YET`. |
| `codex` | `account/rateLimits/read` over `codex app-server`, after the handshake. Spending uses `account/rateLimitResetCredit/consume`. | CLI 0.160.0 or later. |

The Claude request is experimental and the CLI says its shape may change, so the reader
uses only the keys the SDK documents and ignores the rest. The Claude process starts with
no MCP servers, no hooks, no session file and only the user's settings. A reply that
carries a model-turn frame, or a Codex server that announces a thread, stops the read.

## Testing

`agentshim.testing` has fakes that plug into the process fakes:
`ClaudeStreamPeers(account=claude_account(), usage=claude_usage(...))` and
`CodexScript(account=CodexAccount(credits=("c1",)))`. The Codex account is stateful: it
spends the credits it holds and remembers each idempotency key, so a test can check that
a retry spends once.
