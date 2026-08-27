import json
import logging
import os
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from config import SESSIONS_DIR, SYSTEM_PROMPT_TPL, TOOL_REGISTRY

# ------------------------------ Logger Setup ------------------------------
logger = logging.getLogger("agent_core")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# ------------------------------ Harness Sandbox Configuration ------------------------------
WORKSPACE: Path = Path("./workspace").resolve()
INPUT_ROOT: Path = WORKSPACE / "input"
OUTPUT_ROOT: Path = WORKSPACE / "output"

INPUT_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


class HarnessError(Exception):
    """Business-level harness exception. Do NOT expose raw python traceback to agent."""

    pass


def _safe_resolve_input(logical_name: str) -> Path:
    """
    Map logical input filename to workspace/input filesystem path with escape protection.
    """
    p = Path(logical_name)
    if p.is_absolute():
        raise HarnessError(f"Forbidden absolute system path: `{logical_name}`")

    # If already a relative path inside input folder or bare filename
    if (INPUT_ROOT / p).exists():
        target = INPUT_ROOT / p
    elif (WORKSPACE / p).exists():
        target = WORKSPACE / p
    else:
        # Default fallback target inside input directory
        target = INPUT_ROOT / p.name

    real_path = target.resolve()

    # Security check against traversal escape
    try:
        real_path.relative_to(WORKSPACE)
    except ValueError:
        raise HarnessError(
            f"Path traversal escape detected, forbidden name: `{logical_name}`"
        )

    if not real_path.exists():
        raise HarnessError(f"Input file not found in workspace/input: `{logical_name}`")

    return real_path


def _safe_resolve_output(logical_name: str, primary_input_stem: str) -> Path:
    """
    Map logical output path under workspace/output/{primary_input_stem}/{logical_name}.
    """
    p = Path(logical_name)
    if p.is_absolute():
        raise HarnessError(f"Forbidden absolute system path: `{logical_name}`")

    # Create dedicated subfolder under workspace/output/<input_file_stem>/
    base_output_dir = OUTPUT_ROOT / primary_input_stem
    target = base_output_dir / p

    real_path = target.resolve()

    # Security check against traversal escape
    try:
        real_path.relative_to(WORKSPACE)
    except ValueError:
        raise HarnessError(
            f"Path traversal escape detected, forbidden output path: `{logical_name}`"
        )

    return real_path


def _rewrite_tool_arguments(tool_name: str, raw_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Harness argument rewrite layer.
    - Resolves input files under ./workspace/input
    - Routes output targets under ./workspace/output/{input_stem}/{output_path}
    - Ignores non-path options like passwords, flags, and numeric settings
    """
    logger.info(
        f"Harness rewrite tool arguments | tool_name={tool_name}, raw_args={raw_args}"
    )
    
    tool_meta = next((item for item in TOOL_REGISTRY if item["name"] == tool_name), None)
    if not tool_meta:
        raise HarnessError(f"Unknown tool registered: `{tool_name}`")

    rewritten = raw_args.copy()

    # 1. First pass: find primary input file stem for organizing output subfolder
    primary_stem = "default_output"
    raw_inputs = raw_args.get("inputs") or raw_args.get("input") or raw_args.get("file")
    
    if raw_inputs:
        first_file = raw_inputs[0] if isinstance(raw_inputs, list) else raw_inputs
        if isinstance(first_file, str) and first_file.strip():
            primary_stem = Path(first_file).stem

    # 2. Second pass: rewrite parameters explicitly matching registered flags
    for param in tool_meta.get("parameters", []):
        arg_key = param["flag"].lstrip("-")
        val = raw_args.get(arg_key)

        if val is None:
            continue

        param_type = param.get("type", "").lower()
        param_desc = param.get("desc", "").lower()

        # Explicitly skip non-file options (e.g., passwords, credentials, numerical options)
        if any(keyword in arg_key for keyword in ["pass", "password", "key", "token", "limit", "count"]):
            continue

        # Check if parameter specifies input file(s)
        is_input = arg_key in ["inputs", "input", "file", "src"] or "input" in param_desc or "source" in param_desc
        # Check if parameter specifies output destination
        is_output = arg_key in ["output", "out", "dest", "destination"] or "output" in param_desc

        if is_input:
            if isinstance(val, list):
                rewritten[arg_key] = [str(_safe_resolve_input(str(f))) for f in val]
            else:
                rewritten[arg_key] = str(_safe_resolve_input(str(val)))

        elif is_output:
            if isinstance(val, list):
                resolved_list = []
                for out_item in val:
                    target_path = _safe_resolve_output(str(out_item), primary_stem)
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    resolved_list.append(str(target_path))
                rewritten[arg_key] = resolved_list
            else:
                target_path = _safe_resolve_output(str(val), primary_stem)
                # If output specifies a directory name or target file, ensure parent folder structure exists
                target_path.mkdir(parents=True, exist_ok=True) if "." not in target_path.name else target_path.parent.mkdir(parents=True, exist_ok=True)
                rewritten[arg_key] = str(target_path)

    return rewritten


# ---------------- Session disk persistence WITH METADATA ----------------
def get_session_path(sid: str) -> Path:
    return SESSIONS_DIR / f"{sid}.json"


def load_session(sid: str) -> Dict | None:
    """
    Load session from disk.
    Return full session container dict: {"metadata": {...}, "messages": [...]}
    Backward‑compatible: old plain‑message‑only files will be wrapped automatically.
    """
    p = get_session_path(sid)
    logger.debug(f"load_session sid={sid}, path={p}")
    if not p.exists():
        logger.debug(f"load_session sid={sid} not found")
        return None
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    # backward compatibility: legacy file is plain list of messages
    if isinstance(data, list):
        logger.info(f"load_session migrate legacy session {sid}")
        now = datetime.utcnow().isoformat()
        return {
            "metadata": {
                "session_id": sid,
                "created_at": now,
                "updated_at": now,
                "title": "Untitled Chat",
            },
            "messages": data,
        }
    return data


def save_session(sid: str, messages: List[Dict], title: str = None):
    """
    Persist session to disk, auto update updated_at timestamp.
    :param sid: session uuid
    :param messages: message list
    :param title: optional new chat title
    """
    p = get_session_path(sid)
    existing = load_session(sid)
    now_iso = datetime.utcnow().isoformat()
    if existing is None:
        meta = {
            "session_id": sid,
            "created_at": now_iso,
            "updated_at": now_iso,
            "title": title or "Untitled Chat",
        }
    else:
        meta = existing["metadata"]
        meta["updated_at"] = now_iso
        if title is not None:
            meta["title"] = title

    payload = {"metadata": meta, "messages": messages}
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(
        f"save_session sid={sid}, title={meta['title']}, message_count={len(messages)}"
    )


def list_session_ids() -> List[str]:
    ids = []
    for fname in os.listdir(SESSIONS_DIR):
        if fname.endswith(".json"):
            ids.append(fname[:-5])
    return ids


def list_sessions_with_meta() -> List[Dict]:
    """
    List all sessions with metadata for frontend session management.
    Sort by updated_at descending (latest chat first).
    Return list: [{"session_id","created_at","updated_at","title"}, ...]
    """
    out = []
    for sid in list_session_ids():
        container = load_session(sid)
        if container is None:
            continue
        m = container["metadata"]
        out.append(
            {
                "session_id": m["session_id"],
                "created_at": m["created_at"],
                "updated_at": m["updated_at"],
                "title": m["title"],
            }
        )
    # sort latest updated first
    out.sort(key=lambda x: x["updated_at"], reverse=True)
    logger.debug(f"list_sessions_with_meta total={len(out)} sessions")
    return out


def create_new_session() -> tuple[str, List[Dict]]:
    """Create brand‑new chat session, return session_id + plain message list. Metadata saved inside json."""
    # assemble tool description text
    tool_lines = []
    for t in TOOL_REGISTRY:
        param_lines = []
        for p in t["parameters"]:
            req = "required" if p["required"] else "optional"
            param_lines.append(f"  {p['flag']} ({p['type']}, {req}): {p['desc']}")
        tool_lines.append(
            f"Tool name: {t['name']}\nDescription: {t['description']}\nParameters:\n"
            + "\n".join(param_lines)
        )
    tool_list_text = "\n---\n".join(tool_lines)
    #system_content = SYSTEM_PROMPT_TPL.format(tool_list_text=tool_list_text)
    system_content = SYSTEM_PROMPT_TPL.replace("{tool_list_text}", tool_list_text)
    sid = str(uuid.uuid4())
    msg_history = [{"role": "system", "content": system_content}]
    save_session(sid, msg_history, title="Untitled Chat")
    logger.info(f"create_new_session sid={sid}")
    return sid, msg_history


# ---------------- Run external tool (subprocess wrapped by harness) ----------------
def run_tool(tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Harness‑wrapped tool entry point.
    Flow: raw llm args → harness argument rewrite & sandbox validation → subprocess run external script.
    Returns unified result dict, hide raw python traceback from agent observation.
    """
    logger.info(f"run_tool invoke | tool_name={tool_name}, raw_args={tool_args}")
    tool_meta = next(
        (item for item in TOOL_REGISTRY if item["name"] == tool_name), None
    )
    if tool_meta is None:
        err_msg = f"Error: unknown tool `{tool_name}`"
        logger.error(err_msg)
        return {"ok": False, "stdout": "", "stderr": err_msg, "returncode": -1}

    try:
        # Harness: rewrite logical filenames to real filesystem paths, apply sandbox rules
        rewritten_args = _rewrite_tool_arguments(tool_name, tool_args)
    except HarnessError as e:
        err_msg = f"Harness sandbox error: {str(e)}"
        logger.warning(err_msg)
        # Return harness business error, do NOT launch subprocess
        return {"ok": False, "stdout": "", "stderr": err_msg, "returncode": -2}

    cmd = ["python3", tool_meta["script_path"]]
    for param in tool_meta["parameters"]:
        flag = param["flag"]
        key = flag.lstrip("-")
        val = rewritten_args.get(key)
        if param["required"] and val is None:
            err_msg = f"Error: missing required argument {flag}"
            logger.error(err_msg)
            return {"ok": False, "stdout": "", "stderr": err_msg, "returncode": -1}
        if val is None:
            continue
        cmd.append(flag)
        if isinstance(val, list):
            cmd.extend(val)
        else:
            cmd.append(str(val))

    logger.debug(f"run_tool subprocess cmd: {cmd}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    result = {
        "ok": proc.returncode == 0,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "returncode": proc.returncode,
    }
    logger.info(f"run_tool finished tool={tool_name}, returncode={proc.returncode}")
    if not result["ok"]:
        logger.warning(f"run_tool stderr: {proc.stderr}")
    return result
