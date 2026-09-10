"""Private filesystem helper shared by ``schema.py`` and ``mcp.py``."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write(target: Path, content: bytes) -> None:
    """Replace *target* without ever exposing a partially written file.

    The mode of an existing file is preserved: these are config files a user
    may have made read-only or group-writable on purpose. Symlinks are
    followed rather than replaced, because ``rename`` onto a symlink would
    silently turn a link the user set up on purpose into a regular file and
    leave the file it pointed at stale.
    """
    resolved = real_path(target)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    mode = resolved.stat().st_mode if resolved.exists() else None
    handle, temporary_name = tempfile.mkstemp(prefix=f".{resolved.name}.", dir=resolved.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            temporary_path.chmod(mode)
        temporary_path.replace(resolved)
    finally:
        temporary_path.unlink(missing_ok=True)


def real_path(target: Path) -> Path:
    """Resolve every symlink in *target*, existing or not.

    ``Path.resolve()`` would do, but this is the one operation the callers
    need and naming it keeps the reason visible at the call sites: a backup, a
    write and a later delete all have to address the same inode, or restoring
    a config would leave the real file edited and delete only the link.
    """
    return Path(os.path.realpath(target))
