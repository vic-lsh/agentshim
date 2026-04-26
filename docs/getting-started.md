# Getting Started

## Use the Generic Agent Interface

If you want to choose a provider at runtime, instantiate `CodingAgent` directly
with a provider name.

```python
from agentshim import CodingAgent

agent = CodingAgent(provider="claude", model="sonnet")
chat = agent.start_session(cwd=".")

first_reply = chat.generate("Summarize this repository.")
follow_up = chat.generate("Now list the three highest-risk modules.")

print(first_reply)
print(follow_up)
print(chat.session_id)
```

`start_session()` returns a stateful chat object. On the first `generate(...)`
call, `agentshim` starts a fresh provider conversation. On later calls, it
automatically resumes the same underlying provider session using the session id
captured from the first run.

That corresponds roughly to these native CLI flows:

- Claude Code: first call is like `claude -p ...`, later calls add `claude --resume <session_id> ...`
- Codex: first call is like `codex exec ...`, later calls add `codex exec resume <thread_id> ...`
- Gemini: first call is like `gemini ...`, later calls add `gemini --resume <session_id> ...`
- Opencode: first call is like `opencode run ...`, later calls add `opencode run --session <session_id> ...`

## One-Shot Requests

If you only want a one-shot request, use `generate(...)` directly instead of
opening a session.

```python
from agentshim import CodexCodingAgent

agent = CodexCodingAgent(model="gpt-5")
reply = agent.generate("Write a short summary of this codebase.", cwd=".")
print(reply)
```
