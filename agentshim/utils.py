import os
import re
import subprocess
from pathlib import Path
from typing import Any

from loguru import logger


def get_interactive_env() -> dict[str, str]:
    """Capture environment variables from an interactive shell."""
    try:
        # Run env in an interactive shell to get the full user environment
        # Use start_new_session=True (setsid) to detach from TTY and avoid
        # SIGTTOU/SIGTTIN signals when bash -i tries to set process group
        result = subprocess.run(
            ["/bin/bash", "-i", "-c", "env"], capture_output=True, text=True, check=False, start_new_session=True
        )
        if result.returncode != 0:
            return os.environ.copy()

        env: dict[str, str] = {}
        for line in result.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                env[key] = value
        return env
    except Exception:
        return os.environ.copy()


# -- File generation system prompt (shared by RLMCodingAgent & SubagentCodingAgent) --


FILE_GEN_SYSTEM_PROMPT = (
    "You are a deployment assistant. The user will ask you to generate "
    "one or more files. For EACH file, output a section in this exact format:\n\n"
    "FILE: .sds/<filename>\n"
    "```\n"
    "<file content here>\n"
    "```\n\n"
    "Output ONLY these sections. Do not add explanations outside the sections."
)


def generate_and_write_files(
    raw: str,
    prompt: str,
    repo_path: Path,
    log_label: str,
) -> list[str]:
    """Parse ``FILE: .sds/<name>`` sections from *raw* LLM output and write them.

    Returns a list of relative paths that were written.  If no ``FILE:`` sections
    are found but the prompt references exactly one ``.sds/`` file, the entire
    *raw* response (with markdown fences stripped) is written as that file
    (fallback mode).

    Args:
        raw: Raw LLM response text.
        prompt: The original prompt (used to detect expected output files).
        repo_path: Root of the repository.
        log_label: Label for log messages (e.g. ``"[RLM]"``, ``"[Subagent]"``).
    """
    expected_files = re.findall(r"\.sds/[\w._-]+", prompt)

    written: list[str] = []
    file_sections = re.findall(
        r"(?:^|\n)\s*(?:#+\s*)?FILE:\s*(\.sds/[\w./-]+)\s*\n```[^\n]*\n(.*?)```",
        raw,
        re.DOTALL | re.IGNORECASE,
    )
    for rel_path, content in file_sections:
        out_path = repo_path / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content)
        logger.info(f"{log_label} Wrote {out_path}")
        written.append(rel_path)

    if not written and len(expected_files) == 1:
        out_path = repo_path / expected_files[0]
        content = re.sub(r"^```[^\n]*\n|```$", "", raw.strip(), flags=re.MULTILINE)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content)
        logger.info(f"{log_label} Wrote {out_path} (fallback)")
        written.append(expected_files[0])

    return written


DEFAULT_TOOL_PARAM_MAX_LEN = 200
MEMORY_TOOL_PARAM_MAX_LEN = 10000
_VERBOSE_TOOL_NAMES = frozenset({"recall_incident", "store_incident"})


def truncate_params(params: Any, max_len: int = DEFAULT_TOOL_PARAM_MAX_LEN) -> str:
    """Truncate a parameter representation to *max_len* characters."""
    s = str(params)
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def truncate_tool_params(tool_name: str, params: Any) -> str:
    """Render tool parameters with per-tool truncation rules.

    Most tools keep the historical compact preview so CLI logs stay readable.
    Incident-memory tools get a much larger budget because their payloads are
    often the exact debugging context we need to inspect.
    """
    max_len = MEMORY_TOOL_PARAM_MAX_LEN if tool_name in _VERBOSE_TOOL_NAMES else DEFAULT_TOOL_PARAM_MAX_LEN
    return truncate_params(params, max_len=max_len)


def truncate_content(content: str, max_lines: int = 10) -> str:
    """Truncate *content* keeping the first and last *max_lines* lines."""
    lines = content.splitlines()
    if len(lines) > max_lines * 2:
        return "\n".join(lines[:max_lines] + ["... (truncated) ..."] + lines[-max_lines:])
    return content
