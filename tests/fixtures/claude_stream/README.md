# Claude Code stream-json recordings

Four runs of `claude --input-format stream-json --output-format stream-json
--verbose` (Claude Code 2.1.295, model `haiku`, `bypassPermissions`). Each file
is the raw stdout, one JSON object per line. They are scrubbed: the working
directory is `SCRATCH/...`, the account block is `{"scrubbed": true}` and the
user's own commands, skills and agents are samples.

| File | Covers |
| --- | --- |
| `a_two_turns.stdout.jsonl` | `initialize` reply, two turns in one process, `system/init` repeated per turn, cumulative `total_cost_usd` |
| `b_bash_interrupt.stdout.jsonl` | a tool turn, then an interrupted turn (`error_during_execution`, `terminal_reason` `aborted_streaming`), then a third turn that still has the context |
| `c_schema.stdout.jsonl` | `--json-schema`: the enforcement round, a `StructuredOutput` call, `structured_output` on the `result` |
| `d_bad_resume.stdout.jsonl` | a refused `--resume`: one error `result` with `num_turns` 0, then exit 1, before `initialize` is answered |

`tests/unit/providers/claude/test_stream_replay.py` replays them through the
real transport (`ClaudeRecordedPeer`). Re-record them only when the CLI's
stream format changes.
