"""Dedicated CLI homes for ``ConfigScope.PROJECT``.

Some CLIs read the user's global instructions, hooks and memory from their
state root (Codex: ``$CODEX_HOME/AGENTS.md``, ``hooks.json``, ``memories/``)
with no flag to skip them. The only way to keep them out is to run the CLI
against another state root that holds nothing but the credentials. That root
is the caller's to own, because the CLI keeps its conversations there too: a
session resumes only from the home it started in.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

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

    Copies each of ``profile.config_home_files`` that exists in the state
    root *env* selects into *home* (overwriting a stale copy, so a refreshed
    login carries over) and returns the environment overrides that make the
    CLI use *home*. Everything else in *home* (conversations, caches) is the
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
            shutil.copyfile(origin, home / name)
            (home / name).chmod(0o600)
    return {profile.state_root_env: str(home)}
