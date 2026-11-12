"""Runs external tool scripts in isolated subprocesses, harness-wrapped.

Flow: raw LLM args -> harness rewrite + sandbox validation -> subprocess run
(with a timeout) -> sanitized :class:`ToolResult`. Raw tracebacks never escape.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any, Optional

from .harness import HarnessError, rewrite_tool_arguments
from .schemas import ToolResult, ToolSpec, shorten_workspace_paths
from .settings import Settings
from .tool_registry import TOOL_REGISTRY

logger = logging.getLogger("agent.tools")


class ToolRunner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._by_name: dict[str, ToolSpec] = {t.name: t for t in TOOL_REGISTRY}

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._by_name.get(name)

    def _strip_root(self, text: str) -> str:
        """Replace the absolute workspace prefix with ``workspace``.

        The resolved form (e.g. ``/private/tmp/...`` on macOS) must be replaced
        first: the configured form is a prefix of nothing but a *substring* of
        the resolved one, so stripping it first would leave a ``/private`` stub.
        """
        prefixes = {
            str(self.settings.workspace_root),
            str(self.settings.workspace_root.resolve()),
        }
        for prefix in sorted(prefixes, key=len, reverse=True):
            if prefix:
                text = text.replace(prefix, "workspace")
        return shorten_workspace_paths(text)

    def run(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        output_stem: Optional[str] = None,
    ) -> ToolResult:
        logger.info("invoke tool | name=%s args=%s", tool_name, tool_args)
        tool = self._by_name.get(tool_name)
        if tool is None:
            return ToolResult(
                ok=False, stderr=f"Unknown tool `{tool_name}`", returncode=-1
            )

        # 1. Harness: map logical names -> real sandboxed paths.
        try:
            rewritten = rewrite_tool_arguments(tool, tool_args, self.settings, output_stem)
        except HarnessError as exc:
            logger.warning("sandbox error | %s", exc)
            return ToolResult(ok=False, stderr=f"Harness sandbox error: {exc}", returncode=-2)

        # 2. Build argv. Values are passed as separate argv entries (never a
        #    shell string), so there is no shell-injection surface.
        # Use the backend's own interpreter (or TOOL_PYTHON); a bare "python3"
        # may resolve to a system Python without numpy/opencv installed.
        cmd = [self.settings.tool_python, tool.script_path]
        for param in tool.parameters:
            key = param.flag.lstrip("-")
            val = rewritten.get(key)
            if param.required and val is None:
                required = [spec.flag for spec in tool.parameters if spec.required]
                return ToolResult(
                    ok=False,
                    stderr=(
                        "Missing required argument %s (%s). This tool requires: %s. "
                        "Pass every required argument and retry once."
                        % (param.flag, param.desc, ", ".join(required))
                    ),
                    returncode=-1,
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

        # Absolute sandbox prefixes are noise in the UI and in the model
        # context; keep the workspace-relative form only. Both the configured
        # root and its resolved form are stripped, which also covers macOS
        # /tmp -> /private/tmp symlink prefixes.
        out = self._strip_root(proc.stdout or "")
        err = self._strip_root(proc.stderr or "")
        result = ToolResult(
            ok=proc.returncode == 0,
            stdout=out,
            stderr=err,
            returncode=proc.returncode,
        )
        logger.info("tool finished | name=%s returncode=%s", tool_name, proc.returncode)
        if not result.ok:
            logger.warning("tool stderr | %s", result.stderr)
        return result
