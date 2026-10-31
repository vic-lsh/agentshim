# Usage Outside agentshim

A person who uses Claude Code or Codex directly spends tokens agentshim never
sees. Each CLI keeps a transcript of its sessions on disk, and those files are
the only record. `read_transcript_usage` reads the usage out of them.

agentshim keeps no state between calls and has no clock, store or cache of its
own. The caller remembers one `FilePosition` per file, hands them all back, and
gets only what is new. A caller that caches the records also owns the cache.

```python
from datetime import datetime, timedelta, timezone

from agentshim import read_transcript_usage, transcript_roots

since = datetime.now(timezone.utc) - timedelta(days=30)
known = {}                                    # Path -> FilePosition, kept by the caller
for root in transcript_roots("claude", env):
    while True:
        scan = read_transcript_usage(root, known=known, since=since, byte_budget=256 << 20)
        for file in scan.files:
            if file.reset:
                forget_records_of(file.path)       # the file was rewritten
            keep(file.path, file.records)
            known[file.path] = file.position
        if scan.pending == 0:
            break
```

## Where the files are

`transcript_roots(provider, env)` follows each CLI's own rule for its state
directory, from the environment you pass (no process state is read), and
returns directories that may not exist. `transcript_providers()` lists the
providers that can be read.

| Provider | State root | Read |
| --- | --- | --- |
| `claude` | `$CLAUDE_CONFIG_DIR`, else `~/.claude` | `projects/**/*.jsonl` |
| `codex` | `$CODEX_HOME`, else `~/.codex` | `sessions/**/*.jsonl` and `archived_sessions/**/*.jsonl` |

A Codex session file can move from `sessions` to `archived_sessions` after it was
read, leaving a copy in both. `TranscriptRoot.shared_sessions` says so: count
each session from one file only. A `.jsonl.zst` archive is compressed in a way
the standard library of Python 3.10 cannot read, so it is listed in `failed`.

## What a call does

- Files whose modification time is more than 36 hours older than `since` are
  `skipped`. A finished file whose size and time match its known position is
  `unchanged`. Anything else is read.
- A file with a known position is read from where it stopped, if the file did not
  shrink and the 64 bytes before that point still hash the same (FNV-1a).
  Otherwise it is read from the start and `reset` is true: drop everything you
  hold for that path. A rewrite that keeps those 64 bytes and the length
  intact is not detectable.
- `byte_budget` bounds the bytes read per call. The read stops at a line
  boundary, keeps its position, and counts the files left in `pending`. The first
  file read always makes progress, so calling again with the new positions
  finishes for any budget, and a file larger than the budget finishes over
  several calls. The result of any split equals one read of the whole file.
- A line with none of the format's ASCII markers is skipped without decoding it.
  A line over 8 MiB is reduced to the fields the format reads before it is
  decoded, so a huge tool output costs one pass and no memory.
- A file that cannot be read, a directory that cannot be listed and a file the
  format cannot read are in `failed`. They are never treated as empty: keep what
  you hold and report the source as incomplete.
- `FileScan.tail_records` are requests in an unterminated last line, which a
  writer may still be completing. They are provisional: the next call reports
  them in `records` once the line ends. `FileScan.malformed` is the count so far
  over the whole file, not a delta.

## Records

A `TranscriptRecord` is one model request: its timestamp, model, session,
working directory, `TokenUsage`, `UsageSpeed` and `reported_usd`. `reported_usd`
is always `None`: agentshim does not trust a CLI's own cost figure, since an old
CLI bills a model it does not know at a fallback rate. Price the tokens with
`rates_for` and `price_tokens` (see [Usage and Pricing](pricing.md)).

- **Claude.** The CLI writes one line per content block of a message, and each
  repeats the whole usage, so a message is counted once by `message.id` and
  `requestId`. A parser remembers the last 64 keys, which keeps its state small;
  a repeat further back counts again. Cache writes are split by TTL from
  `cache_creation`. Speed is `usage.speed`. An advisor call, which runs on its own
  model, is its own record, read from `usage.iterations`. A Task subagent writes
  its own file under `<session>/subagents/`, and its records take the session id
  from that path, so they belong to the parent. Messages the CLI wrote itself
  (`<synthetic>`) are skipped.
- **Codex.** A `token_count` line carries no model or tier, so the parser carries
  forward the model of the latest `turn_context` and the tier of the latest
  `thread_settings_applied`. The first `session_meta` names the session and its
  directory. Each request is told from a repeat by the thread's running totals;
  the copied history at the start of a forked or subagent thread is a baseline,
  not usage. The service tier a rollout does not state reads as standard.

The formats are read from the CLIs' session files and public documentation, and a
CLI release can change them. A format change is an agentshim release.

## Adding a format

A provider plugin that implements `transcripts()` (the optional `TranscriptSource`
capability) returns a `TranscriptFormat`: where its files are, the ASCII markers
that can start a useful line, the fields a huge line is reduced to, and a parser
factory. The parser takes one line at a time, returns `TranscriptRecord`s, counts
what it cannot use, and saves its state as bytes at a line boundary. A parser
that cannot read a saved state returns `None`, and the file is read again whole.
