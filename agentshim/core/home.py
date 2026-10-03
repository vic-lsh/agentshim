"""Dedicated CLI homes for ``ConfigScope.PROJECT``.

Some CLIs read the user's global instructions, hooks and memory from their
state root (Codex: ``$CODEX_HOME/AGENTS.md``, ``hooks.json``, ``memories/``)
with no flag to skip them. The only way to keep them out is to run the CLI
against another state root that holds nothing but the credentials. That root
is the caller's to own, because the CLI keeps its conversations there too: a
session resumes only from the home it started in.

The credentials are linked, not copied. Codex rotates its OAuth refresh token
on refresh (the server rejects a reused one with ``refresh_token_reused``) and
saves ``auth.json`` by opening it with ``O_TRUNC`` and writing in place
(codex-rs ``login/src/auth/storage.rs``, ``FileAuthStorage::save``), which
follows a symlink. A copy would strand one of the two logins after the first
refresh; a link keeps a single file that every process rereads before it
refreshes (``AuthManager::refresh_token``'s guarded reload).
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from ._files import real_path

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .profile import ProviderProfile


def state_root(profile: ProviderProfile, env: Mapping[str, str]) -> Path | None:
    """The state root the CLI run with *env* reads, or ``None`` if it has none.

    ``$<state_root_env>`` when set, else ``$HOME/<state_dirs[0]>``.
    """
    if profile.state_root_env and env.get(profile.state_root_env):
        return Path(env[profile.state_root_env]).expanduser()
    if not profile.state_dirs:
        return None
    home = env.get("HOME") or os.path.expanduser("~")  # noqa: PTH111 - a str contract, not a Path
    return Path(home) / profile.state_dirs[0]


def prepare_config_home(
    profile: ProviderProfile, home: Path, env: Mapping[str, str]
) -> dict[str, str]:
    """Make *home* a state root holding only the credentials, and point at it.

    Links each of ``profile.config_home_files`` that exists in the state root
    *env* selects into *home* as a symlink to that file's real path (replacing
    whatever was there, such as a copy from an older agentshim), so a token
    the CLI refreshes in either home is the one both read next. Returns the
    environment overrides that make the CLI use *home*. A sandbox that runs
    the CLI must expose the link's target read-write. Everything else in *home* (conversations, caches) is the
    CLI's own and is left alone. A provider with no ``config_home_files``
    isolates by flags alone: nothing is written and the result is empty.

    Raises:
        ValueError: *home* is not absolute, or is the state root *env*
            already selects, which would isolate nothing.
    """
    if not profile.config_home_files:
        return {}
    if not home.is_absolute():
        msg = f"{profile.name} config home must be an absolute path, got {home}"
        raise ValueError(msg)
    assert profile.state_root_env is not None, "config_home_files needs a state_root_env"  # noqa: S101 - profile invariant
    source = state_root(profile, env)
    if source is not None and real_path(source) == real_path(home):
        msg = f"{profile.name} config home {home} is the user's own state root"
        raise ValueError(msg)
    home.mkdir(parents=True, exist_ok=True)
    for name in profile.config_home_files:
        origin = None if source is None else source / name
        if origin is not None and origin.is_file():
            _link(home / name, real_path(origin))
    return {profile.state_root_env: str(home)}


def _link(link: Path, target: Path) -> None:
    """Point *link* at *target*, atomically replacing any file already there.

    Safe to call concurrently for the same *link* from threads or processes:
    each call stages its own uniquely named link, and ``replace`` is atomic, so
    the last writer wins with the same target. A link that already points at
    *target* is success, whoever made it.
    """
    if link.is_symlink() and link.readlink() == target:
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    staged = link.with_name(
        f".{link.name}.{os.getpid()}.{threading.get_ident()}.{uuid4().hex}.link"
    )
    staged.symlink_to(target)
    try:
        staged.replace(link)
    except OSError:
        staged.unlink(missing_ok=True)
        if link.is_symlink() and link.readlink() == target:
            return
        raise
