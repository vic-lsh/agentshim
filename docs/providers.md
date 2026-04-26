# Providers

If you already know which backend you want, construct the provider class
yourself.

```python
from agentshim import ClaudeCodeCodingAgent

agent = ClaudeCodeCodingAgent(model="sonnet")
chat = agent.start_session(cwd=".")

print(chat.generate("What does this project do?"))
print(chat.generate("Which files should I read first?"))
```

The bundled provider classes are:

- `ClaudeCodeCodingAgent`
- `CodexCodingAgent`
- `GeminiCodingAgent`
- `OpencodeCodingAgent`

Use `CodingAgent(provider=...)` when the provider should be selected at runtime.
