"""A minimal stdio MCP server with one tool, ``whoami``, that answers with a marker.

Usage: ``python fake_mcp_server.py <marker>``. The e2e tests seed one into a
configuration the CLI would load by default and give another to the session,
then check which markers the model could reach.
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    marker = sys.argv[1]
    for line in sys.stdin:
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = message.get("method")
        if "id" not in message:
            continue
        if method == "initialize":
            result = {
                "protocolVersion": message["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "1"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "whoami",
                        "description": "Return this server's marker string.",
                        "inputSchema": {"type": "object", "properties": {}},
                    }
                ]
            }
        elif method == "tools/call":
            result = {"content": [{"type": "text", "text": marker}]}
        else:
            result = {}
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}))
        sys.stdout.write("\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
