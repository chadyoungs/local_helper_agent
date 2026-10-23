"""Runs external tool scripts in isolated subprocesses, harness-wrapped.

Flow: raw LLM args -> harness rewrite + sandbox validation -> subprocess run
(with a timeout) -> sanitized :class:`ToolResult`. Raw tracebacks never escape.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any

from .harness import HarnessError, rewrite_tool_arguments
from .schemas import ToolResult, ToolSpec
from .settings import Settings
from .tool_registry import TOOL_REGISTRY

logger = logging.getLogger("agent.tools")


class ToolRunner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._by_name: dict[str, ToolSpec] = {t.name: t for t in TOOL_REGISTRY}

    def get(self, name: str) -> ToolSpec | None:
        return self._by_name.get(name)

    def run(self, tool_name: str, tool_args: dict[str, Any]) -> ToolResult:
        logger.info("invoke tool | name=%s args=%s", tool_name, tool_args)
        tool = self._by_name.get(tool_name)
        if tool is None:
            return ToolResult(
                ok=False, stderr=f"Unknown tool `{tool_name}`", returncode=-1
            )

        # 1. Harness: map logical names -> real sandboxed paths.
        try:
            rewritten = rewrite_tool_arguments(tool, tool_args, self.settings)
        except HarnessError as exc:
            logger.warning("sandbox error | %s", exc)
            return ToolResult(ok=False, stderr=f"Harness sandbox error: {exc}", returncode=-2)

        # 2. Build argv. Values are passed as separate argv entries (never a
        #    shell string), so there is no shell-injection surface.
        cmd = ["python3", tool.script_path]
        for param in tool.parameters:
            key = param.flag.lstrip("-")
            val = rewritten.get(key)
            if param.required and val is None:
                return ToolResult(
                    ok=False, stderr=f"Missing required argument {param.flag}", returncode=-1
                )
            if val is None:
                continue
            cmd.append(param.flag)
            if isinstance(val, list):
                cmd.extend(str(v) for v in val)
            else:
                cmd.append(str(val))

        logger.debug("subprocess cmd: %s", cmd)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.settings.tool_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            logger.warning("tool timed out | %s", tool_name)
            return ToolResult(
                ok=False,
                stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
                stderr=f"Tool `{tool_name}` timed out after {self.settings.tool_timeout_seconds}s",
                returncode=-3,
                timed_out=True,
            )
        except OSError as exc:
            logger.error("tool failed to start | %s err=%s", tool_name, exc)
            return ToolResult(ok=False, stderr=f"Failed to start tool: {exc}", returncode=-4)

        result = ToolResult(
            ok=proc.returncode == 0,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            returncode=proc.returncode,
        )
        logger.info("tool finished | name=%s returncode=%s", tool_name, proc.returncode)
        if not result.ok:
            logger.warning("tool stderr | %s", result.stderr)
        return result
