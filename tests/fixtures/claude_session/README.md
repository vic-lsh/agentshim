# Claude Code session file, synthetic

`streamed_message.jsonl` is made up. No line is copied from a real session, and
every text, path, id and number is invented. Its shape follows what Claude Code
writes to `projects/<project>/<session>.jsonl`:

- One `assistant` line per content block of a message, all sharing `message.id`
  and `requestId`, in a row.
- The prompt side of the usage is the same on every line of a message, but
  `output_tokens` is what the stream had reached when the line was written, so
  it grows: `8, 8, 152` for the first message and `6, 6, 6, 311` for the last.
  Most messages write the finished count on every line (the second one: `40, 40`).
- `speed` and `iterations` appear on the lines written once the usage was
  complete, not on the earlier ones.
- A message the CLI wrote itself has the model `<synthetic>` and bills nothing.
- `user` lines, which hold no usage, sit between the messages.

Tests that read this file must get the usage of each message once, at its
finished count: output `152 + 40 + 311 = 503`, three requests.
