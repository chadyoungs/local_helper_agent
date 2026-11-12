"""Harness sandbox: maps logical tool arguments to real filesystem paths.

Security model
--------------
The LLM only ever emits *logical* filenames (e.g. ``2016.pdf``, ``result/out.png``).
This module resolves them to paths under the workspace, blocking:

- absolute system paths (``/home/...``, ``C:\\...``)
- path-traversal escapes (``../`` reaching outside the workspace)

Missing input files raise a clean :class:`HarnessError` -- no raw Python
traceback ever reaches the LLM or the frontend.

Input files may live in sub-folders of ``workspace/input``; they are addressed
by their relative path (``2024/scan.jpg``), exactly as the frontend lists them.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .schemas import ToolSpec
from .settings import Settings


class HarnessError(Exception):
    """Business-level sandbox error. Message is safe to show the user/LLM."""


def _relative_name(path: Path, root: Path) -> str:
    """POSIX-style path of ``path`` relative to ``root`` (for messages/lists)."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


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


def _fuzzy_candidates(query: str, root: Path) -> list[Path]:
    """Return input files that loosely match ``query`` (for long filenames).

    Matching ignores whitespace, ``_``, ``-`` and ``.``, so a partial or
    abbreviated name (e.g. ``2016 毕`` -> ``2016届毕业典礼.pdf``) can still
    resolve to a unique file. Sub-folders are searched too, so ``scan`` finds
    ``2024/scan.jpg``. Order is by relative path for stable ambiguity lists.
    """
    q = re.sub(r"[\s_\-\.]+", "", query.lower())
    if not q:
        return []
    out: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file() or p.name == ".gitkeep":
            continue
        rel = _relative_name(p, root)
        norm = re.sub(r"[\s_\-\.]+", "", rel.lower())
        if norm == q or norm.startswith(q) or q in norm:
            out.append(p)
    return sorted(out, key=lambda p: _relative_name(p, root).lower())


def list_input_files(settings: Settings) -> list[dict[str, Any]]:
    """Every file under ``workspace/input``, recursively, as relative paths.

    The frontend renders this list, and the same relative name is what the
    harness accepts as a logical input filename.
    """
    root = settings.input_root
    files: list[dict[str, Any]] = []
    if not root.exists():
        return files
    for p in root.rglob("*"):
        if not p.is_file() or p.name == ".gitkeep":
            continue
        rel = _relative_name(p, root)
        try:
            size = p.stat().st_size
        except OSError:  # file vanished between listing and stat
            continue
        files.append(
            {
                "name": rel,
                "dir": rel.rsplit("/", 1)[0] if "/" in rel else "",
                "size": size,
            }
        )
    files.sort(key=lambda f: f["name"].lower())
    return files


def resolve_input(logical_name: str, settings: Settings) -> Path:
    """Resolve a logical input filename to a real, existing path in input_root.

    Only logical/relative names under ``workspace/input`` are accepted. Absolute
    paths and path traversal are always blocked by the sandbox.

    If the exact name is missing, a unique fuzzy match (partial / abbreviated
    filename) is accepted; ambiguous matches raise with the candidate list so
    the agent can pick the right file.
    """
    root = settings.input_root
    real = _is_absolute_or_escaping(logical_name, root)
    if real.exists() and real.is_file():
        return real

    candidates = _fuzzy_candidates(logical_name, root)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        names = ", ".join(_relative_name(p, root) for p in candidates)
        raise HarnessError(
            f"Ambiguous file `{logical_name}`; matches: {names}. "
            f"Please specify which one you mean."
        )
    raise HarnessError(
        f"Input file not found in workspace/input: `{logical_name}`. "
        + _input_listing(root)
    )


def _input_listing(root: Path, limit: int = 12) -> str:
    """Short "available files" hint so the model corrects its guess in one retry.

    Without it a small local model keeps inventing plausible filenames, burns
    the whole step budget on failed calls, and the user only sees the generic
    "I exceeded the step budget" message.
    """
    try:
        files = sorted(
            p for p in root.rglob("*") if p.is_file() and p.name != ".gitkeep"
        )
    except OSError:
        return ""
    if not files:
        return "workspace/input is empty -- ask the user to add the files first."
    names = [_relative_name(p, root) for p in files[:limit]]
    hint = "Available files: " + ", ".join(names)
    if len(files) > limit:
        hint += " (+%d more)" % (len(files) - limit)
    hint += ". Use one of those names verbatim; never invent a filename."
    return hint


def build_output_dir_name(title: str, when: Optional[datetime] = None) -> str:
    """Build a sanitized ``workspace/output`` sub-folder name from chat + time.

    Example: ``合并毕业照`` at 14:30:05 on 2026-10-24 ->
    ``合并毕业照_20261024_143005``.
    """
    when = when or datetime.now()
    safe = re.sub(r"[^\w\u4e00-\u9fff\-]+", "_", title.strip())
    safe = re.sub(r"_+", "_", safe).strip("_")[:40]
    if not safe:
        safe = "chat"
    return f"{safe}_{when.strftime('%Y%m%d_%H%M%S')}"


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
    tool: ToolSpec,
    raw_args: dict[str, Any],
    settings: Settings,
    output_stem: Optional[str] = None,
) -> dict[str, Any]:
    """Harness argument rewrite layer.

    - input  params -> absolute existing paths (list preserved or joined)
    - output params -> absolute paths under workspace/output/<primary_stem>
    - value  params -> passed through untouched

    ``output_stem`` overrides the folder under ``workspace/output`` (default:
    derived from the first input filename). It is normally the per-turn
    ``<chat title>_<timestamp>`` name supplied by the API layer.

    For ``pdf_merge``, the ``inputs`` param is always kept as ONE string
    (space-joined) because that is what its CLI expects.
    """
    rewritten: dict[str, Any] = dict(raw_args)
    primary_stem = output_stem or _first_input_stem(tool, raw_args)

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
