# Events

Every turn emits a stream of frozen event dataclasses. `AgentEvent` is their
union.

| Event | Fields | Meaning |
|---|---|---|
| `RunStarted` | `argv` | the CLI process is about to start |
| `RunFinished` | `exit_code` | the CLI process exited |
| `SessionStarted` | `session_id` | the provider named the conversation |
| `AssistantText` | `text`, `item_id`, `agent` | assistant-facing message text |
| `Reasoning` | `text`, `item_id`, `agent` | thinking or reasoning text |
| `TextDelta` | `text`, `kind`, `item_id`, `agent` | a fragment of an item still being written (stream transport, opt in) |
| `ToolCall` | `tool_id`, `tool`, `args`, `agent` | a tool was invoked |
| `ToolResult` | `tool_id`, `tool`, `stdout`, `stderr`, `exit_code`, `duration_s`, `agent` | a tool finished |
| `SkillsDiscovered` | `names` | the provider listed the skills it offers |
| `SkillInvoked` | `name`, `source_path`, `tool_id` | the agent loaded a skill |
| `UsageReport` | `usage`, `cost_usd` | provider accounting |
| `Lifecycle` | `kind`, `detail` | provider plumbing |
| `Stderr` | `text` | one stderr line |
| `RawOutput` | `text` | one stdout line that was not a provider event |
| `ProviderError` | `message` | the provider reported an error |
| `TurnInterrupted` | none | the turn was interrupted before it finished; the conversation is kept |
| `ApprovalDenied` | `kind`, `detail` | the agent asked for permission or input and the `ApprovalPolicy` refused |
| `RateLimitStatus` | `window`, `used_fraction`, `resets_at`, `limit`, `window_minutes`, `exhausted`, `raw` | the provider reported where one rate-limit window stands (optional, see below) |

## Handling them

A handler is anything with `on_event(event)`.

```python
from agentshim import AssistantText, CliAgent, EventHandlerBase, ToolCall

class Watcher(EventHandlerBase):
    def on_event(self, event):
        if isinstance(event, ToolCall):
            print("tool:", event.tool)
        elif isinstance(event, AssistantText):
            print(event.text, end="")

agent = CliAgent("claude", event_handler=Watcher())
```

`event_handlers=[...]` takes several; passing both spellings composes them in
order. `ConsoleEventHandler` renders to any text stream, `NullEventHandler`
drops everything, and `CompositeEventHandler` fans out explicitly.

```python
import sys
from agentshim import CliAgent, ConsoleEventHandler

# Watcher is the handler defined above.
agent = CliAgent("claude", event_handlers=[ConsoleEventHandler(sys.stderr), Watcher()])
```

## Thread contract

`on_event` always runs on the thread that called `AgentSession.turn()`. The
executor reads the CLI's pipes on helper threads but drains them on the
calling thread, so a handler needs no locking of its own and an exception it
raises propagates out of `turn()` after the process is killed.

## Tool results

A tool that failed is reported the same way on every provider: the message on
`ToolResult.stderr` with a nonzero `exit_code`, and `stdout` empty. A
renderer can therefore branch on `exit_code` alone and never mistake a
failure for success, which is what `ConsoleEventHandler` does: a failed
result is red, a successful one green. A provider that reports a real exit
code passes it through; one that only reports a boolean failure uses `1`.

## Skills

A provider reports skills through two events. `SkillsDiscovered` carries the
skills offered to the session, once per run, when the stream lists them.
`SkillInvoked` is emitted where the agent loaded a skill, directly after the
`ToolCall` that loaded it (its `tool_id` names that call). How each CLI shows
a load is provider knowledge and stays inside agentshim; a caller never
matches tool names or paths.

`TurnResult.skills` is a `SkillSummary` folded from the same events by
`SkillTracker`, which is an ordinary handler: register your own
`SkillTracker(profile)` to summarize several turns.

| `SkillSummary` | Meaning |
|---|---|
| `discovered` | offered skill names, or `None` when the stream did not list them |
| `invocations` | the `SkillInvoked` events, or `None` when the provider cannot reveal loads |
| `invoked` | distinct invoked names in first-use order, or `None` |
| `invocation_count` | number of loads, or `None` |

`None` means unknown, never zero. `profile.skill_discovery` and
`profile.skill_invocation` declare the signal ahead of a turn as a
`SkillSignal`: `NONE` (unknown), `STRUCTURED` (a dedicated CLI frame) or
`INFERRED` (derived from tool activity, so a load by other means can be
missed). See [Providers](providers.md) for the per-provider matrix.

Which skills are offered at all is a session option:
`start_session(skill_scope=SkillScope.PROJECT)` limits it to the workspace's
skills (see [Providers](providers.md)).

## Rate limits

`RateLimitStatus` is the provider's own report of one rate-limit window, so a
caller can pause before a limit is hit instead of learning of it from a failed
turn. It is optional: a provider that has no such signal emits nothing, and a
field the provider did not state is `None`, never zero. One event describes
one window, so a provider with several windows emits several events; keep the
newest per `(limit, window)`.

| Field | Meaning |
|---|---|
| `window` | the window in the provider's words (`five_hour`, `seven_day`, `primary`, `secondary`) |
| `limit` | the limit it belongs to when there is more than one (Codex `limitId`) |
| `used_fraction` | share used, `0.0` to `1.0` (`remaining_fraction` is its complement) |
| `resets_at` | epoch seconds when the window resets |
| `window_minutes` | window length when the provider states it |
| `exhausted` | `True` reached, `False` not reached, `None` not stated |
| `raw` | the provider's own mapping, for diagnostics |

| Provider | Source | Emitted |
|---|---|---|
| Claude (stream transport and one-shot) | `rate_limit_event` | once per `unifiedWindows` entry; `exhausted` follows `status` for the window named by `rateLimitType` |
| Codex (app-server transport) | `account/rateLimits/updated` | once per reported `primary` / `secondary` window, also between turns; `exhausted` is true when `rateLimitReachedType` or `spendControlReached` is set |
| Others | none | never |
## Observing a conversation

These are additive, for a caller that renders a conversation as it happens. They are
metadata and opt-ins: `Session.run` returns the same turn with or without them, and the
metadata fields do not take part in equality, so `AssistantText("x", "id", "agent")` equals
`AssistantText("x")`.

- **`item_id`** names the item an event belongs to, stably, so a replay names it the same.
  Claude: `"<message id>:<content block index>"` for text and reasoning, and the `tool_use`
  id for tools. Codex: the thread item id. `None` where the provider gives nothing to name.
- **`agent`** names the subagent that produced the event: the `tool_use` id of the call that
  started it (Claude's `parent_tool_use_id`). `None` is the lead. Codex does not attribute:
  its child threads' notifications are dropped.
- **`TextDelta`** (`partial_text=True` on `Agent.session`) streams the fragments of an item
  before its `AssistantText`, `Reasoning` or `ToolResult` arrives. For one `item_id` the
  deltas, joined, are a prefix of the text the completed event carries. Claude adds
  `--include-partial-messages`; Codex reads `item/agentMessage/delta`,
  `item/reasoning/summaryTextDelta` and `item/commandExecution/outputDelta`. A one-shot
  transport refuses the option.
- **`raw_sink`** receives a `RawFrame(direction, text)` for every line crossing the
  provider's pipe, in order, one line without its newline. Recorded frames replay through
  `ClaudeFramePeer`.
- **`disable_subagents=True`** starts the provider without its own delegation tools: Claude
  gets `--disallowedTools` for `Agent`, `Task` and the tools that reach other Claude Code
  sessions (`SendMessage`, `ListAgents`, `SendToProject`, `SendToProjectContext`,
  `RemoteTrigger`, `Workflow`); Codex gets `--disable multi_agent`, which the CLI
  refuses when the name is unknown, so a rename fails at startup.
- **`disable_self_scheduling=True`** starts the provider without the tools it would schedule its
  own later turns with, for a caller that starts every turn itself: a wake-up the agent
  arranged would never fire. Claude gets `--disallowedTools` for `ScheduleWakeup`, `CronCreate`,
  `CronDelete` and `CronList`, in the same list as `disable_subagents` when both are set. Codex
  has no such tools and ignores it. A one-shot transport refuses the option.
- **`new_conversation_id`** chooses the id of a new conversation (Claude `--session-id`). Codex
  names threads itself and refuses it. `Session.warm()` opens the conversation before the first
  turn and returns its id (Codex's from `thread/start`).
- **`Session.poll_idle(on_event)`** delivers what the provider said between turns, which a
  turn would otherwise read late or discard, and returns `False` once the provider's process
  has ended. Call it while no turn runs; it never blocks.

## Steering

`Session.steer` reports through three events, emitted on the thread running the
turn: `SteerDelivered(text)` when the provider accepted the message,
`SteerConsumed(text)` when it entered the turn, and `SteerRejected(text, reason)`
when the provider refused it. See [Sessions](sessions.md).

## Usage

Every provider emits at least one `UsageReport` per turn when its CLI reports
usage, and the last one matches `TurnResult.usage`. `ProviderUsage.tokens`
is the normalized breakdown in [Usage and Pricing](pricing.md): total input
including cache reads and writes, the two cache classes, output including
reasoning, and the reasoning part.
`ProviderUsage.raw` keeps the CLI's own mapping for diagnostics.

Copilot CLI prints no token counts at all, so its counts are zero; see
[Providers](providers.md).
