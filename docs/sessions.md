# Sessions and transports

An agent run that lasts hours meets every failure a conversation can have:
the provider is overloaded, the conversation it was resuming is gone, a
Codex thread has grown too slow, the process restarted. agentshim splits
that into two tiers so each stays small.

| Tier | Type | Owns | Does not do |
|---|---|---|---|
| Conversation | `Conversation`, opened by a `Transport` | one provider conversation over one live connection | retry, restart, wait, decide |
| Session | `Session`, made by `Agent.session` | one logical conversation across failures | talk to a provider |

## Using a session

```python
from agentshim import Agent, ApprovalPolicy, NativePermissions, TurnRequest

agent = Agent(
    "claude",
    permissions=NativePermissions.bypass(),   # no defaults: the caller decides
    approvals=ApprovalPolicy.DENY,
)
with agent.session("/work/repo") as session:
    ticket = session.prepare_turn(TurnRequest(prompt="Fix the failing test"))
    journal(ticket.turn_id)                   # the id exists before anything runs
    turn = session.run(ticket)
    print(turn.result.text, turn.continuity)
```

`prepare_turn` reserves the turn and names it (`ticket.turn_id`, from the
injected `IdAllocator`), so a caller can journal its intent before the model
is called. `run(ticket)` executes it. A ticket runs once, and only the most
recent ticket runs; a stale or reused one raises `SessionStateError`.
`interrupt()` is thread-safe: it ends the running turn (the result has
`interrupted=True` and the conversation is kept) and also ends a retry wait.

Waiting and time go through the injected `Clock` (`clock.wait(seconds, stop)`),
never `time.sleep`, so tests with `FakeClock` and `SequentialIds` run with no
real waiting.

## Continuity

`Turn.continuity` is judged against the conversation the session held when
the turn was prepared (`ticket.expected_conversation`):

| Value | Meaning |
|---|---|
| `CONTINUED` | the turn ran in the expected conversation (or the session had none yet) and it is retained |
| `RESET` | it ran in the expected conversation, which was retired afterwards; the next turn starts cold |
| `REPLACED` | it did not run in the expected conversation: a fresh one replaced it during the turn |

A caller that shortens a prompt because the conversation already carries its
instructions needs both facts: `session.conversation_id` is where the **next**
turn continues (`None` after a reset), and `session.last_turn_conversation_id`
is where the **last** turn ran (a reset keeps it).

Two prepare options make continuity a requirement:

* `expect_conversation="conv-1"` demands exactly that conversation. If the
  session holds another one, or none, `prepare_turn` raises `ContinuityError`.
  The turn also never retries fresh and is exempt from renewal.
* `pin=True` keeps the conversation through renewal and never retries a
  refused resume in a fresh conversation. Use it for work whose history must
  survive.

## Recovery rules

These are implemented by `SessionPolicy` in `core/session_policy.py`, a pure
step function `step(state, event) -> (state, commands)`. The shell only
carries out the commands (`Open`, `Execute`, `Wait`, `CloseConversation`,
`SaveCheckpoint`, `ClearCheckpoint`, `Finish`, `Fail`, `Refuse`).

1. **Transient retry.** A `FailureKind.TRANSIENT` turn failure is retried in
   place, in the same conversation, after each delay of
   `RetryPolicy.delays` (default `(30, 60, 120, 240, 480)` seconds). After the
   last delay the error propagates. `interrupt()` or `close()` during a wait
   ends it and raises the transient error. Every other kind propagates at once.
2. **Resume refused.** When a turn that continued a conversation raises
   `SessionResumeError`, the conversation is dropped and the turn is retried
   once in a fresh one (`REPLACED`). A second failure propagates. Strict and
   pinned turns never retry fresh.
3. **Backstop.** A resumed turn that fails with `FailureKind.OTHER` forgets its
   conversation before the error propagates, so the next turn starts fresh.
   Classified failures (`TRANSIENT` after its retries, `USAGE_LIMIT`, `AUTH`,
   `SCHEMA`), timeouts and interrupts keep it. Strict and pinned turns keep it
   too.
4. **Renewal.** After a successful turn, if the profile's `RenewalBudget` is
   reached (`max_turns` successful turns in the conversation, or one turn with
   `max_turn_input_tokens` input tokens or `max_turn_duration_ms`), the
   conversation is retired (`RESET`). Only Codex declares a budget today.
5. **Idle release.** With `idle_release_after=seconds`, a live conversation
   that has been idle at least that long is closed at prepare time and reopened
   by id before the turn; continuity stays `CONTINUED`. `session.release()`
   does it on demand.
6. **Checkpoints.** With a `CheckpointStore` and a `checkpoint_key`, the session
   loads the saved `Checkpoint(conversation_id, usage)` on start and resumes
   it, saves it after each turn whenever a conversation is retained, and clears
   it when the conversation is retired, replaced or forgotten.
   `resume_id=` (with `previous_usage=`) wins over a stored checkpoint.
   `session.adopt(id)` offers a conversation later and returns `False` if the
   session already holds newer history, a turn is prepared or running, or the
   provider cannot resume.

## Writing a Transport

A transport is the one place that knows how to reach a provider. Implement the
two protocols in `agentshim.core.conversation`:

```python
class MyTransport:
    @property
    def profile(self) -> ProviderProfile: ...   # declare native_permission_modes, supports_resume, renewal
    def open(self, spec: ConversationSpec) -> Conversation: ...

class MyConversation:
    @property
    def conversation_id(self) -> str | None: ...
    def turn(self, request, emit): ...          # run one turn; call emit(event) on the calling thread
    def interrupt(self): ...                    # thread-safe; the running turn returns interrupted=True
    def close(self): ...                        # idempotent
```

Rules a transport follows:

* **Never retry or recover.** Raise `TurnFailedError` with the right
  `FailureKind`, `SessionResumeError` for a conversation that is gone (at `open`
  or at the first turn), `TurnTimeoutError` for a timeout. The session decides
  what happens next, and catches only these base types.
* **Reject unsupported permissions.** `open` raises `ProviderCapabilityError` for
  a `spec.permissions.mode` that is not in `profile.native_permission_modes`.
* **Honor `ConversationSpec`.** `resume_id` and `previous_usage` start from an
  existing conversation; `mcp_servers` and `reasoning_effort` are fixed for the
  conversation's life.
* **Interrupt means "end this turn".** `interrupt` returns a result with
  `interrupted=True` (and emits `TurnInterrupted`) and keeps the conversation;
  with no turn running it does nothing.

Subclass `TransportContract` and `ConversationContract` from
`agentshim.testing.contracts` as `Test<Impl>` and implement the factories; pytest
then runs the inherited checks against your transport. `FakeTransport` is the
scripted double for testing code that drives sessions:

```python
transport = FakeTransport([turn_failed(FailureKind.TRANSIENT), FakeTurn(text="done")])
agent = Agent(transport, permissions=..., approvals=..., clock=FakeClock(), ids=SequentialIds())
```

## Confinement

`Agent(..., confinement=c)` wraps the executor with `confine(executor, c)`, uses
`c.env` as the agent environment (so `env=` may not also be given), and expresses
the working directory, absolute stdio-MCP paths and a schema directory the way
the agent sees them (`c.agent_path`).

## The one-shot transport

`Agent("claude", ...)` reaches the provider through `OneShotTransport`, which
runs one CLI process per turn exactly as `CliAgent` / `AgentSession` do and
supports `NativeMode.BYPASS` only. It is transitional: the long-lived transports
replace it, and the scope arguments of `Agent.session` go with it.

## The Claude stream transport

`Agent("claude", transport=TransportKind.STREAM, ...)` (or
`ClaudeStreamTransport(executor=..., clock=..., ids=...)` passed to `Agent`)
keeps one `claude --input-format stream-json --output-format stream-json
--verbose` process per conversation. `open` spawns it, sends `initialize` and
waits for the reply, so a refused `--resume` surfaces there as
`SessionResumeError` (the CLI writes one error `result` and exits before it
answers). A turn writes one user message and reads until that turn's `result`
frame; the process keeps its context between turns.

| Concern | Behaviour |
| --- | --- |
| Permissions | `BYPASS` is `--permission-mode bypassPermissions`. `WORKSPACE_WRITE` is `acceptEdits` plus `--add-dir` for each writable root, Claude's OS sandbox for bash (writable: working directory and roots; `failIfUnavailable`; no unsandboxed escape; no network) and a `PreToolUse` hook denying `Edit`/`Write`/`NotebookEdit` outside those roots. `READ_ONLY` and `network=True` raise `ProviderCapabilityError`: Claude's sandbox always grants the working directory and temp directories to commands, so "read but not write" cannot be promised. |
| Approvals | The process runs with `--permission-prompts none`, so the CLI denies anything that would prompt and lists it in the result's `permission_denials`. Those become `ApprovalDenied` events; with `FAIL_TURN` the turn then fails. A `can_use_tool` request that arrives anyway is denied at once (`FAIL_TURN`: with `interrupt: true`, then the turn fails); `hook_callback`, `mcp_message` and unknown requests get an error answer. The transport never leaves a request unanswered. |
| Schema, MCP servers, effort | These are flags of the process, but a schema is chosen per turn. A turn whose set differs from the running process's restarts the process with `--resume=<id>`; the conversation id is unchanged, so the session reports `CONTINUED`. A fresh conversation whose first turn carries a schema therefore costs one extra process start. |
| Cost and usage | `total_cost_usd` is cumulative over the process; the turn's cost is the difference from the previous result (reset by a restart). `usage` is per turn. |
| Errors | An API failure arrives as a `result` with subtype `success` and `is_error` true; every `is_error` result is classified (`TRANSIENT`, `USAGE_LIMIT`, `AUTH`, `SCHEMA`, `OTHER`) by the same rules as the one-shot path. A process that dies mid-turn raises `CliExitError` (a `TurnFailedError`) with its stderr; the next turn resumes in a new process. |
| Interrupt | `interrupt()` sends a control request; the turn's `result` is `error_during_execution` with `terminal_reason` `aborted_*`, which becomes `TurnResult.interrupted` and `TurnInterrupted`. The process and conversation survive. |
| Timeouts | `TurnRequest.timeout` is measured on the injected `Clock`. An overrun interrupts the turn, waits `interrupt_grace_s` for its result, then raises `TurnTimeoutError` (killing the process if it ignored the interrupt; the next turn resumes). |
| Close | stdin is closed, then terminate, then kill, each after `close_grace_s`. |

`CLAUDECODE` is removed from the child's environment, and Claude Code's own
system prompt is kept. The working directory is fixed at spawn: a turn with a
different `cwd` raises `ProviderCapabilityError`. Test code scripts the CLI with
`ClaudeStreamPeers` (`FakeExecutor([], peers=peers.build)`) and replays real
recordings with `ClaudeRecordedPeer`.
