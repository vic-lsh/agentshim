# Codex usage fixtures

The existing `exec_gpt6_luna.jsonl` and skill fixtures are recorded CLI stdout.
Their usage mappings are unchanged.

`resume_vibesys_totals.jsonl` is a minimal reconstruction of the two input
counts supplied in the VibeSys bug report: 2,019,100 followed by 12,564,014
for the same thread. The thread id is synthetic. Output, cache and reasoning
counts were not supplied, so these fields are omitted rather than invented.
This is not a newly recorded CLI run. Codex 0.160.0 was found on the
interactive shell's PATH, but the requested `gpt-6.1-sol` / medium probe
failed to initialize its app-server because the unchanged Codex home is
read-only in this sandbox. It emitted no JSON events. The repository
contained no two-invocation recording.

Codex 0.144.4's JSON event processor copies all five fields from the thread
usage `total` into `turn.completed.usage`; it does not expose a per-turn field:
[provider source](https://github.com/openai/codex/blob/rust-v0.144.4/codex-rs/exec/src/event_processor_with_jsonl_output.rs).
