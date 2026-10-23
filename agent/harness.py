"""Harness sandbox: maps logical tool arguments to real filesystem paths.

Security model
--------------
The LLM only ever emits *logical* filenames (e.g. ``2016.pdf``, ``result/out.png``).
This module resolves them to paths under the workspace, blocking:

- absolute system paths (``/home/...``, ``C:\\...``)
- path-traversal escapes (``../`` reaching outside the workspace)

Missing input files raise a clean :class:`HarnessError` -- no raw Python
traceback ever reaches the LLM or the frontend.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .schemas import ToolSpec
from .settings import Settings


class HarnessError(Exception):
    """Business-level sandbox error. Message is safe to show the user/LLM."""


def _is_absolute_or_escaping(logical_name: str, root: Path) -> Path:
    """Reject absolute paths and return the resolved in-root path or raise."""
    p = Path(logical_name)
    if p.is_absolute():
        raise HarnessError(f"Forbidden absolute system path: `{logical_name}`")
    real = root.joinpath(p).resolve()
    try:
        real.relative_to(root.resolve())
    except ValueError:
        raise HarnessError(f"Path traversal escape detected: `{logical_name}`")
    return real


def resolve_input(logical_name: str, settings: Settings) -> Path:
    """Resolve a logical input filename to a real, existing path in input_root."""
    root = settings.input_root
    real = _is_absolute_or_escaping(logical_name, root)
    if not real.exists():
        raise HarnessError(f"Input file not found in workspace/input: `{logical_name}`")
    return real


def resolve_output(logical_name: str, settings: Settings, primary_stem: str = "default_output") -> Path:
    """Resolve a logical output path under workspace/output/<primary_stem>/."""
    base = settings.output_root / primary_stem
    real = _is_absolute_or_escaping(logical_name, base)
    return real


def split_logical_names(value: str) -> list[str]:
    """Split a space/comma-separated list of logical filenames."""
    return [f.strip() for f in re.split(r"[,\s]+", value) if f.strip()]


def _first_input_stem(tool: ToolSpec, raw_args: dict[str, Any]) -> str:
    """Pick a stable folder name under output/ from the first input filename."""
    for key in tool.input_param_keys:
        val = raw_args.get(key)
        if not val:
            continue
        if isinstance(val, str):
            parts = split_logical_names(val)
            first = parts[0] if parts else ""
        elif isinstance(val, list) and val:
            first = str(val[0])
        else:
            first = ""
        if first:
            return Path(first).stem
    return "default_output"


def rewrite_tool_arguments(
    tool: ToolSpec, raw_args: dict[str, Any], settings: Settings
) -> dict[str, Any]:
    """Harness argument rewrite layer.

    - input  params -> absolute existing paths (list preserved or joined)
    - output params -> absolute paths under workspace/output/<primary_stem>
    - value  params -> passed through untouched

    For ``pdf_merge``, the ``inputs`` param is always kept as ONE string
    (space-joined) because that is what its CLI expects.
    """
    rewritten: dict[str, Any] = dict(raw_args)
    primary_stem = _first_input_stem(tool, raw_args)

    # pdf_merge special: normalise a stray list back into a space-joined string.
    if tool.name == "pdf_merge" and "inputs" in rewritten:
        if isinstance(rewritten["inputs"], list):
            rewritten["inputs"] = " ".join(str(f) for f in rewritten["inputs"])

    for param in tool.parameters:
        key = param.flag.lstrip("-")
        val = rewritten.get(key)
        if val is None:
            continue

        if param.kind == "input":
            if param.multi:
                if isinstance(val, str):
                    names = split_logical_names(val)
                elif isinstance(val, list):
                    names = [str(f) for f in val]
                else:
                    names = [str(val)]
                resolved = [str(resolve_input(name, settings)) for name in names]
                # pdf_merge expects a single space-joined string of real paths.
                if tool.name == "pdf_merge":
                    rewritten[key] = " ".join(resolved)
                else:
                    rewritten[key] = resolved
            else:
                rewritten[key] = str(resolve_input(str(val), settings))

        elif param.kind == "output":
            if param.multi:
                resolved_out = []
                for item in val if isinstance(val, list) else [val]:
                    target = resolve_output(str(item), settings, primary_stem)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    resolved_out.append(str(target))
                rewritten[key] = resolved_out
            else:
                target = resolve_output(str(val), settings, primary_stem)
                if "." in target.name:  # a file target -> ensure parent dir
                    target.parent.mkdir(parents=True, exist_ok=True)
                else:  # a folder target -> ensure the folder exists
                    target.mkdir(parents=True, exist_ok=True)
                rewritten[key] = str(target)

        # kind == "value": leave unchanged (passwords, counts, colors, ...)

    return rewritten
