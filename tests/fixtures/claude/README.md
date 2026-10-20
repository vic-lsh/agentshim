# Claude Code stream-json recordings

Real `claude -p --output-format stream-json --verbose` stdout. The `init`
frame's lists of slash commands, skills, plugins, agents and MCP servers are
emptied, and local paths are replaced with `/workspace`; every other frame is
verbatim.

| File | Covers |
| --- | --- |
| `haiku45_two_model_calls.jsonl` | Claude Code 2.1.304, `claude-haiku-4-5`: a `Read` call, then the answer, so two model calls in one turn |
| `haiku45_prompt_too_long_blocking_limit.jsonl` | Claude Code 2.1.304, `claude-haiku-4-5`: a prompt over the CLI's own blocking limit; compaction fails, no request is sent, `terminal_reason: blocking_limit` |
| `haiku45_prompt_too_long_api_400.jsonl` | Claude Code 2.1.304, `claude-haiku-4-5`: a prompt the CLI underestimated (random CJK text), refused by the API with a 400, `terminal_reason: prompt_too_long` |
| `haiku_task_subagent.jsonl`, `skill_*_haiku.jsonl`, `sonnet_1h_cache.jsonl` | older recordings: a Task subagent, skill loads, one-hour cache writes |
