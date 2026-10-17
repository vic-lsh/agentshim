"""Static, versioned per-model facts: each model's context window and where it was read.

The table is data: one :class:`ModelSpec` per ``(vendor, model)``, each with
the official page its numbers were read from and the date they were checked,
and a table-level ``version`` and ``last_updated``. It is keyed by vendor, the
same way as the pricing table, because a context window belongs to the model,
whichever CLI runs it.

A lookup of a model the table does not know returns ``None``: an unknown
window is never guessed from a similar name, a prefix or a model family. A
bracketed variant (``claude-opus-5-5[1m]``) also returns ``None``, because a
variant may change the window. Callers pin or extend assumptions by passing
their own :class:`ModelTable` (see :meth:`ModelTable.with_entries`).

Which models a provider CLI is known to run, and which of them is its
default, is per-provider data on ``ProviderProfile.models``.

Updating the table is described in ``docs/models.md``.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, cast

from agentshim.core.pricing import PROVIDER_VENDORS

if TYPE_CHECKING:
    from collections.abc import Iterable

_DATE_SUFFIX = re.compile(r"-\d{8}$")

#: Vendor whose models each provider name, or ``vendor/`` prefix, runs: every
#: one pricing knows, plus the prefixes opencode names Google's models with
#: (``google/``, ``google-vertex/``). Google is not priced, so pricing's map
#: lacks them.
MODEL_VENDORS: dict[str, str] = {
    **PROVIDER_VENDORS,
    "google": "google",
    "google-vertex": "google",
}

OPENAI_MODELS_URL = "https://developers.openai.com/api/docs/models"
GOOGLE_MODELS_URL = "https://ai.google.dev/gemini-api/docs/models"
ANTHROPIC_CONTEXT_WINDOWS_URL = (
    "https://platform.claude.com/docs/en/build-with-claude/context-windows"
)


@dataclass(frozen=True)
class ModelSpec:
    """One model's context window, with where and when it was read.

    Attributes:
        vendor: Who serves the model (``"openai"``, ``"anthropic"``).
        model: The model id as the vendor names it.
        context_window: The model's context window in tokens (input and
            output together), at the vendor's default for that id.
        source: URL of the official page the number was read from.
        checked: Date the number was last checked against ``source``.
        display_name: A readable name for the model (``"Opus 5.5"``), or
            ``None`` for a caller-built entry that has none.
        family: The model line the model belongs to (``"Claude"``, ``"GPT"``,
            ``"Gemini"``); set together with ``display_name``.
        tier: The model's rank within its family, ``1`` the smallest and
            cheapest; versions of one tier share a rank. ``None`` for a model
            that sits on no tier ladder.
        tier_name: The tier's readable name (``"Opus"``); set together with
            ``tier``.
    """

    vendor: str
    model: str
    context_window: int
    source: str
    checked: dt.date
    display_name: str | None = None
    family: str | None = None
    tier: int | None = None
    tier_name: str | None = None

    def __post_init__(self) -> None:
        """Reject a missing name, source or date, and a window that is not a positive int."""
        for name in ("vendor", "model"):
            value: object = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                msg = f"{name} must be a non-empty string"
                raise ValueError(msg)
        source = cast("object", self.source)
        if not isinstance(source, str) or not source.startswith("https://"):
            msg = f"{self.vendor}/{self.model}: source must be an https:// URL"
            raise ValueError(msg)
        checked = cast("object", self.checked)
        if not isinstance(checked, dt.date) or isinstance(checked, dt.datetime):
            msg = f"{self.vendor}/{self.model}: checked must be a datetime.date"
            raise TypeError(msg)
        window = cast("object", self.context_window)
        if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
            msg = f"{self.vendor}/{self.model}: context_window must be a positive int"
            raise ValueError(msg)
        self._check_presentation()

    def _check_presentation(self) -> None:
        """Reject a half-set name pair or tier pair, and a tier that is not a positive int."""
        for first, second in (("display_name", "family"), ("tier", "tier_name")):
            if (getattr(self, first) is None) != (getattr(self, second) is None):
                msg = f"{self.vendor}/{self.model}: {first} and {second} are set together"
                raise ValueError(msg)
        for name in ("display_name", "family", "tier_name"):
            value: object = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                msg = f"{self.vendor}/{self.model}: {name} must be a non-empty string"
                raise ValueError(msg)
        tier = cast("object", self.tier)
        if tier is not None and (isinstance(tier, bool) or not isinstance(tier, int) or tier < 1):
            msg = f"{self.vendor}/{self.model}: tier must be a positive int"
            raise ValueError(msg)
        if self.tier is not None and self.family is None:
            msg = f"{self.vendor}/{self.model}: tier needs a family"
            raise ValueError(msg)

    @property
    def key(self) -> tuple[str, str]:
        """The ``(vendor, model)`` this entry describes."""
        return (self.vendor, self.model)


@dataclass(frozen=True)
class ModelTable:
    """A versioned set of :class:`ModelSpec` entries.

    Attributes:
        version: Names this set of facts; change it whenever a fact changes.
        last_updated: Date the table as a whole was last updated. No entry
            may have been checked later.
        entries: One entry per ``(vendor, model)``.
    """

    version: str
    last_updated: dt.date
    entries: tuple[ModelSpec, ...] = field(default=())

    def __post_init__(self) -> None:
        """Reject a missing version, duplicate keys, and entries checked after ``last_updated``."""
        version = cast("object", self.version)
        if not isinstance(version, str) or not version.strip():
            msg = "version must be a non-empty string"
            raise ValueError(msg)
        object.__setattr__(self, "entries", tuple(self.entries))
        seen: set[tuple[str, str]] = set()
        for entry in self.entries:
            if not isinstance(cast("object", entry), ModelSpec):
                msg = f"entries must be ModelSpec, got {entry!r}"
                raise TypeError(msg)
            if entry.key in seen:
                msg = f"duplicate model entry for {entry.vendor}/{entry.model}"
                raise ValueError(msg)
            seen.add(entry.key)
            if entry.checked > self.last_updated:
                msg = (
                    f"{entry.vendor}/{entry.model} was checked {entry.checked}, "
                    f"after the table's last_updated {self.last_updated}"
                )
                raise ValueError(msg)

    def get(self, vendor: str, model: str) -> ModelSpec | None:
        """The entry for exactly ``(vendor, model)``, or ``None``."""
        for entry in self.entries:
            if entry.key == (vendor, model):
                return entry
        return None

    def with_entries(
        self,
        entries: Iterable[ModelSpec],
        *,
        version: str,
        last_updated: dt.date | None = None,
    ) -> ModelTable:
        """A new table with *entries* added, replacing any with the same key."""
        added = tuple(entries)
        by_key = {entry.key: entry for entry in self.entries}
        by_key.update({entry.key: entry for entry in added})
        latest = max([self.last_updated, *(entry.checked for entry in added)])
        return replace(
            self,
            version=version,
            last_updated=last_updated or latest,
            entries=tuple(by_key.values()),
        )


def model_spec(
    provider: str, model: str | None, table: ModelTable | None = None
) -> ModelSpec | None:
    """The facts about *model* run through *provider*, or ``None`` when they are not known.

    Accepts a provider name (``codex``, ``claude``) or a vendor name, and a
    model with a ``vendor/`` prefix (opencode's ``provider/model``). Drops a
    trailing date snapshot (``claude-haiku-4-5-20251001``). Returns ``None``
    for a bracketed variant (``claude-opus-5-5[1m]``), for a provider that
    runs no known vendor's models, and for any model the table lacks; it
    never matches by prefix or family.
    """
    if not model:
        return None
    name = model.strip().lower()
    if "[" in name:
        return None
    vendor = MODEL_VENDORS.get(provider.strip().lower())
    if "/" in name:
        prefix, _, rest = name.partition("/")
        vendor = MODEL_VENDORS.get(prefix)
        name = rest
    if vendor is None or not name:
        return None
    return (table or DEFAULT_MODELS).get(vendor, _DATE_SUFFIX.sub("", name))


def context_window(provider: str, model: str | None, table: ModelTable | None = None) -> int | None:
    """*model*'s context window in tokens when run through *provider*, or ``None`` when not known."""
    spec = model_spec(provider, model, table)
    return None if spec is None else spec.context_window


def default_models() -> ModelTable:
    """The table agentshim ships (:data:`DEFAULT_MODELS`)."""
    return DEFAULT_MODELS


_CHECKED = dt.date(2026, 10, 17)


def _openai(
    model: str,
    context_window: int,
    display_name: str,
    tier: tuple[int, str] | None = None,
) -> ModelSpec:
    # Each model's page under OPENAI_MODELS_URL ("<n> context window").
    return ModelSpec(
        vendor="openai",
        model=model,
        context_window=context_window,
        source=f"{OPENAI_MODELS_URL}/{model}",
        checked=_CHECKED,
        display_name=display_name,
        family="GPT",
        tier=None if tier is None else tier[0],
        tier_name=None if tier is None else tier[1],
    )


def _anthropic(model: str, context_window: int, tier: tuple[int, str], version: str) -> ModelSpec:
    # "Context window sizes by model": the 1M models list 1M as the default
    # with no beta header; "other Claude models ... have a 200k-token" window.
    return ModelSpec(
        vendor="anthropic",
        model=model,
        context_window=context_window,
        source=ANTHROPIC_CONTEXT_WINDOWS_URL,
        checked=_CHECKED,
        display_name=f"{tier[1]} {version}",
        family="Claude",
        tier=tier[0],
        tier_name=tier[1],
    )


def _google(model: str, context_window: int, display_name: str, tier: tuple[int, str]) -> ModelSpec:
    # Each model's page under GOOGLE_MODELS_URL ("Input token limit"), which
    # is the window Google publishes for the model.
    return ModelSpec(
        vendor="google",
        model=model,
        context_window=context_window,
        source=f"{GOOGLE_MODELS_URL}/{model}",
        checked=_CHECKED,
        display_name=display_name,
        family="Gemini",
        tier=tier[0],
        tier_name=tier[1],
    )


# Tier ladders, smallest and cheapest first. Anthropic ranks its lines
# Haiku < Sonnet < Opus < Fable; Codex's GPT lines Luna < Terra < Sol < Astra.
_HAIKU = (1, "Haiku")
_SONNET = (2, "Sonnet")
_OPUS = (3, "Opus")
_FABLE = (4, "Fable")
_LUNA = (1, "Luna")
_TERRA = (2, "Terra")
_SOL = (3, "Sol")
_ASTRA = (4, "Astra")
_PRO = (1, "Pro")

_ONE_MILLION = 1_000_000
_GOOGLE_1M = 1_048_576
_OPENAI_1_05M = 1_050_000

#: The table agentshim ships. Update it as described in ``docs/models.md``.
#: A model whose window could not be read from its vendor's page is left out.
DEFAULT_MODELS = ModelTable(
    version="2026-10-17.1",
    last_updated=_CHECKED,
    entries=(
        # OpenAI (Codex): each model page reads "1,050,000 context window".
        _openai("gpt-6-astra", _OPENAI_1_05M, "GPT-6 Astra", _ASTRA),
        _openai("gpt-6.1-sol", _OPENAI_1_05M, "GPT-6.1 Sol", _SOL),
        _openai("gpt-6-sol", _OPENAI_1_05M, "GPT-6 Sol", _SOL),
        _openai("gpt-6-luna", _OPENAI_1_05M, "GPT-6 Luna", _LUNA),
        _openai("gpt-5.6-sol", _OPENAI_1_05M, "GPT-5.6 Sol", _SOL),
        _openai("gpt-5.6-terra", _OPENAI_1_05M, "GPT-5.6 Terra", _TERRA),
        _openai("gpt-5.6-luna", _OPENAI_1_05M, "GPT-5.6 Luna", _LUNA),
        _openai("gpt-5.5", _OPENAI_1_05M, "GPT-5.5"),
        # Anthropic (Claude).
        _anthropic("claude-fable-5-1", _ONE_MILLION, _FABLE, "5.1"),
        _anthropic("claude-fable-5", _ONE_MILLION, _FABLE, "5"),
        _anthropic("claude-opus-5-5", _ONE_MILLION, _OPUS, "5.5"),
        _anthropic("claude-opus-5", _ONE_MILLION, _OPUS, "5"),
        _anthropic("claude-opus-4-8", _ONE_MILLION, _OPUS, "4.8"),
        _anthropic("claude-opus-4-7", _ONE_MILLION, _OPUS, "4.7"),
        _anthropic("claude-opus-4-6", _ONE_MILLION, _OPUS, "4.6"),
        _anthropic("claude-sonnet-5-5", _ONE_MILLION, _SONNET, "5.5"),
        _anthropic("claude-sonnet-5", _ONE_MILLION, _SONNET, "5"),
        _anthropic("claude-sonnet-4-6", _ONE_MILLION, _SONNET, "4.6"),
        _anthropic("claude-haiku-5-5", _ONE_MILLION, _HAIKU, "5.5"),
        _anthropic("claude-haiku-4-5", 200_000, _HAIKU, "4.5"),
        # Google (opencode): the page reads "Input token limit 1,048,576".
        _google("gemini-3-pro-preview", _GOOGLE_1M, "Gemini 3 Pro Preview", _PRO),
    ),
)
