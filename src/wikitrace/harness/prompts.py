"""Jinja2 prompt loader: a role is a directory, a filename is the condition."""

# Each owning package calls `register_prompt(role, dir)` at import; a missing context key raises (`StrictUndefined`).

from __future__ import annotations

import hashlib
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

# The package root, so the run fingerprint covers exactly the shipped .j2 files.
_ROOT = Path(__file__).resolve().parent.parent

REGISTRY: dict[str, str] = {}

_ENV = Environment(
    loader=FileSystemLoader(str(_ROOT)),
    undefined=StrictUndefined,
    autoescape=False,
    keep_trailing_newline=True,
    trim_blocks=False,
    lstrip_blocks=False,
)


def register_prompt(role: str, directory: str) -> None:
    """Bind a prompt role to the directory holding its templates."""
    REGISTRY[role] = directory.rstrip("/")


def _dir(role: str) -> str:
    if role not in REGISTRY:
        raise KeyError(f"Unknown prompt role: {role!r}")
    return REGISTRY[role]


def render_system(
    role: str, variant: str | None = None, *, suffix: str = "", **context
) -> str:
    """Render `<role dir>/[<variant>.]system[<suffix>].j2` for this role's condition."""
    # `variant` names the condition; `suffix` selects a prompt FAMILY where one role has two.
    stem = f"{variant}.system" if variant else "system"
    return _ENV.get_template(f"{_dir(role)}/{stem}{suffix}.j2").render(**context).strip("\n")


def render_user(
    role: str, variant: str | None = None, outcome: str | None = None,
    *, suffix: str = "", **context,
) -> str:
    """Render `<role dir>/[<variant>.]user[.<outcome>][<suffix>].j2`, substituting `context`."""
    stem = ".".join(p for p in (variant, "user", outcome) if p)
    return _ENV.get_template(f"{_dir(role)}/{stem}{suffix}.j2").render(**context).strip("\n")


def render_template(name: str, **context) -> str:
    """Render a template by its path under the package root, for a prompt that is no role's."""
    # One template including the solver's own partials, so the excerpt cannot drift from what the solver read.
    return _ENV.get_template(name).render(**context).strip("\n")


def replace_each(text: str, edits: tuple[tuple[str, str], ...]) -> str:
    """``text`` with each ``(old, new)`` applied, refusing an ``old`` not found exactly once."""
    for old, new in edits:
        if (n := text.count(old)) != 1:
            raise ValueError(f"expected one occurrence of {old!r}, found {n}")
        text = text.replace(old, new)
    return text


def system_prompt(role: str) -> str:
    """The role's default system prompt; system prompts carry no variables."""
    return render_system(role)


def prompts_source_hash() -> str:
    """Stable sha256 over every `.j2` template's path + bytes."""
    # Feeds the run fingerprint, so an edit to any prompt makes a new cohort id even before it is committed.
    h = hashlib.sha256()
    for p in sorted(_ROOT.rglob("*.j2")):
        h.update(p.relative_to(_ROOT).as_posix().encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()
