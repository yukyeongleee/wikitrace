"""Domain imports and resources use the canonical package layout."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import re
import subprocess
import sys


def test_registry_resolves_each_benchmark_to_its_domain_adapter():
    from wikitrace.envs import resolve

    for domain, benchmark in (("textgame", "alfworld"), ("textgame", "scienceworld"),
                              ("textgame", "plancraft"), ("tau3", "tau3"),
                              ("appworld", "appworld")):
        assert type(resolve(domain, benchmark)).__module__ == f"wikitrace.envs.{domain}.adapter"


def test_imports_work_in_a_fresh_process():
    code = """
from wikitrace.envs import resolve
from wikitrace.config import make_split
from wikitrace.run.build import build
from wikitrace.run.cell import run_cell
assert callable(build) and callable(run_cell) and callable(make_split)
assert resolve("textgame", "alfworld").benchmark is not None
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_run_layer_imports_no_single_domain():
    """The shared layer binds to the registry, never to one domain's modules."""
    package = Path(import_module("wikitrace").__file__).resolve().parent
    offenders = []
    for path in sorted((package / "run").rglob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.match(r"\s*(from|import)\s+wikitrace\.envs\.\w", line) and "appworld.workflow" not in line:
                offenders.append(f"{path.relative_to(package)}:{number}: {line.strip()}")
    assert not offenders, offenders


# What each layer may import at module level.
_LAYERS = ("config", "harness", "memory", "envs", "run", "analysis", "cli")
_ALLOWED = {layer: set(_LAYERS[: i + 1]) for i, layer in enumerate(_LAYERS)}


def _layer_of(module: str) -> str:
    """`wikitrace.memory.store` -> `memory`; a top-level module is its own layer."""
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else ""


def _module_level_imports(tree) -> list[tuple[int, str]]:
    """Every `wikitrace.X` a module names at module level, `if TYPE_CHECKING` included."""

    # A type-only import still names the layer it is written against; only nested imports are skipped.
    import ast

    found: list[tuple[int, str]] = []

    def visit(body):
        for node in body:
            if isinstance(node, ast.Import):
                found.extend((node.lineno, a.name) for a in node.names
                             if a.name.split(".")[0] == "wikitrace")
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] == "wikitrace":
                    found.append((node.lineno, node.module))
            elif isinstance(node, ast.If):
                visit(node.body)
                visit(node.orelse)
            elif isinstance(node, ast.Try):
                for part in (node.body, node.orelse, node.finalbody):
                    visit(part)
                for handler in node.handlers:
                    visit(handler.body)
    visit(tree.body)
    return found


def test_packages_import_only_downwards():
    """config <- harness <- memory <- envs <- run <- cli, by the AST."""
    import ast

    root = Path(import_module("wikitrace").__file__).resolve().parent
    offenders = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        importer = relative.parts[0] if len(relative.parts) > 1 else relative.stem
        allowed = _ALLOWED.get(importer)
        if allowed is None:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for number, target in _module_level_imports(tree):
            layer = _layer_of(target)
            if layer and layer not in allowed:
                offenders.append(f"{relative}:{number}: {importer} imports {target}")
    assert not offenders, offenders
