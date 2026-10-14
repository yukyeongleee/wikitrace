"""Frozen suite manifests: committed JSON read back as an explicit-role split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent / "data"


def read_manifest(name: str) -> dict[str, Any]:
    """The committed ``data/<name>.manifest.json`` (the generator under scripts/ writes it)."""
    path = DATA_DIR / f"{name}.manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing — run scripts/setup/build_{name}_manifest.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def roled(manifest: dict[str, Any], train_key: str, test_key: str) -> list[tuple[dict, str]]:
    """``[(entry, role), ...]`` in manifest order: the train list, then the test list."""
    # Order is the split's contract: a head slice (`limit_explicit_split`) must span the task types or bins.
    return [(e, "train") for e in manifest[train_key]] + [(e, "test") for e in manifest[test_key]]


def interpreter(root: Path, hint: str) -> str:
    """The benchmark venv's python under ``root``, or a loud pointer at the installer."""
    python = root / ".venv" / "bin" / "python"
    if not python.exists():
        raise FileNotFoundError(f"{python} is missing — run {hint}")
    return str(python)
