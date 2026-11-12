"""Guards for the "Python 3.9 and newer" promise.

Two things can silently break 3.9 support:

1. syntax that only exists in 3.10+ (``match``, ``X | Y`` used directly,
   parenthesised context managers with more than one item),
2. PEP 604 unions in *annotations* -- they parse everywhere but explode at
   runtime on 3.9 when something evaluates them (Pydantic, FastAPI, dataclasses,
   ``typing.get_type_hints``).

Both are checked here without needing a 3.9 interpreter.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {
    "__pycache__", ".git", ".pytest_cache", ".venv", "venv",
    "workspace", "sessions", "_test_ws", "_test_ws2",
}


def _py_files():
    out = []
    for path in PROJECT.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.relative_to(PROJECT).parts):
            continue
        out.append(path)
    return sorted(out)


def _annotations(node):
    """Every annotation expression attached to ``node``."""
    found = []
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = node.args
        for arg in list(args.args) + list(args.kwonlyargs) + list(getattr(args, "posonlyargs", [])):
            if arg.annotation is not None:
                found.append(arg.annotation)
        for extra in (args.vararg, args.kwarg):
            if extra is not None and extra.annotation is not None:
                found.append(extra.annotation)
        if node.returns is not None:
            found.append(node.returns)
    elif isinstance(node, ast.AnnAssign):
        found.append(node.annotation)
    return found


def _is_pep604(annotation) -> bool:
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        return True
    # Optional[...] / Union[...] are fine; report only the "|" form.
    return False


@pytest.mark.parametrize("path", _py_files(), ids=lambda p: p.relative_to(PROJECT).as_posix())
def test_module_parses_with_python39_grammar(path):
    source = path.read_text(encoding="utf-8")
    try:
        ast.parse(source, filename=str(path), feature_version=(3, 9))
    except SyntaxError as exc:  # pragma: no cover - failure path
        pytest.fail("%s is not valid Python 3.9 syntax: %s" % (path.name, exc))


def test_no_pep604_union_in_annotations():
    offenders = []
    for path in _py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            for annotation in _annotations(node):
                for sub in ast.walk(annotation):
                    if _is_pep604(sub):
                        offenders.append("%s:%s" % (path.relative_to(PROJECT), sub.lineno))
    assert not offenders, (
        "PEP 604 unions in annotations break Python 3.9 when they are evaluated; "
        "use Optional[str] / Union[...] instead: " + ", ".join(offenders)
    )
