# Codex app-server transcripts

Three recorded sessions of `codex app-server` (codex-cli 0.160.0, model
`gpt-6-luna`, low effort) over stdio. Each line is
`{"dir": "out" | "in", "msg": {...}}`: `out` is what the client wrote, `in` what
the server wrote. They are scrubbed: the Codex home is `$CODEX_HOME`, the
working directory `$CWD`, and the install id, host name and credit balance are
placeholders.

| File | Covers |
| --- | --- |
| `session_1.jsonl` | handshake, `thread/resume` errors (unknown and malformed id), two threads in one process, a plain turn |
| `session_2.jsonl` | `workspace-write` with `untrusted` approvals: a declined command approval, a command that runs, `turn/interrupt` |
| `session_3.jsonl` | `thread/resume` across processes, a structured-output turn, per-thread MCP config that fails to start, a failed turn (unknown model) |

`tests/unit/providers/codex/app_server/test_replay.py` parses every line.
Re-record them only when the CLI version in
`scripts/codex_protocol/schema.json` changes.
