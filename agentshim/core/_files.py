"""Private filesystem helper shared by ``schema.py`` and ``mcp.py``."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write(target: Path, content: bytes) -> None:
    """Replace *target* without ever exposing a partially written file.

    The mode of an existing file is preserved: these are config files a user
    may have made read-only or group-writable on purpose.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = target.stat().st_mode if target.exists() else None
    handle, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            temporary_path.chmod(mode)
        temporary_path.replace(target)
    finally:
        temporary_path.unlink(missing_ok=True)
