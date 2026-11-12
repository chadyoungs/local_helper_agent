"""Pydantic models shared across the agent.

These are the single source of truth for (a) how the model's structured output
is validated and (b) how tools are described to the LLM.
"""

from __future__ import annotations

import re

from typing import Any, Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Tool specification
# --------------------------------------------------------------------------- #
class ParamSpec(BaseModel):
    """A single CLI flag a tool accepts.

    ``kind`` tells the harness how to treat the value during path rewriting:
      - "input":  a logical input filename (mapped into workspace/input)
      - "output": a logical output path (mapped into workspace/output)
      - "value":  a plain value (password, count, color, ...) passed through
    """

    flag: str
    type: Literal["str", "int", "list[str]"] = "str"
    required: bool = False
    desc: str = ""
    kind: Literal["input", "output", "value"] = "value"
    multi: bool = False  # True if the value is a list of strings


class ToolSpec(BaseModel):
    """Declarative description of an external tool script."""

    name: str
    description: str
    script_path: str
    parameters: list[ParamSpec] = Field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Schema helpers
    # ------------------------------------------------------------------ #
    @property
    def input_param_keys(self) -> list[str]:
        return [p.flag.lstrip("-") for p in self.parameters if p.kind == "input"]

    @property
    def output_param_keys(self) -> list[str]:
        return [p.flag.lstrip("-") for p in self.parameters if p.kind == "output"]

    def llm_schema(self) -> dict[str, Any]:
        """Return the tool in Ollama function-calling schema form (if enabled)."""
        properties: dict[str, Any] = {}
        required: list[str] = []
        for p in self.parameters:
            key = p.flag.lstrip("-")
            js_type = "array" if p.multi else ("integer" if p.type == "int" else "string")
            prop: dict[str, Any] = {"type": js_type, "description": p.desc}
            if p.multi:
                prop["items"] = {"type": "string"}
            properties[key] = prop
            if p.required:
                required.append(key)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def cli_schema_text(self) -> str:
        """Human-readable parameter list used in the system prompt."""
        lines = []
        for p in self.parameters:
            req = "required" if p.required else "optional"
            lines.append(f"  {p.flag} ({p.type}, {req}, kind={p.kind}): {p.desc}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Agent decision & tool result
# --------------------------------------------------------------------------- #
class AgentDecision(BaseModel):
    """Structured decision produced by the LLM each agent-loop step.

    Exactly one of ``need_ask_user``, ``tool_name`` or ``final_answer`` should
    be "active" per decision; the agent loop interprets them in that order.
    """

    need_ask_user: bool = False
    question: str = ""

    tool_name: str = ""
    tool_args: dict[str, Any] = Field(default_factory=dict)

    final_answer: str = ""

    @property
    def is_question(self) -> bool:
        return self.need_ask_user

    @property
    def is_tool_call(self) -> bool:
        return bool(self.tool_name) and not self.need_ask_user

    @property
    def is_final(self) -> bool:
        return bool(self.final_answer) and not self.need_ask_user and not self.tool_name


_WORKSPACE_PATH_RE = re.compile(r"[^\s\"',\[\]]*?/workspace/(input|output)/")


def shorten_workspace_paths(text: str) -> str:
    """Drop the absolute prefix of sandbox paths: ``.../workspace/input/a.pdf``
    becomes ``workspace/input/a.pdf``. Keeps tool output readable for the model
    and for the UI without losing where the file actually is.
    """
    return _WORKSPACE_PATH_RE.sub(r"workspace/\1/", text)


class ToolResult(BaseModel):
    """Unified, sanitized result of running an external tool."""

    ok: bool = False
    stdout: str = ""
    stderr: str = ""
    returncode: int = -1
    timed_out: bool = False

    def summary(self, max_chars: int = 2000) -> str:
        """Compact, model-facing text: a verdict first, then a one-paragraph digest.

        Two things made small local models loop until the step budget ran out:
        the raw JSON dump buried the success/failure signal, and the model could
        not tell whether the previous call had already done the job. So the
        verdict comes first and an explicit "do not repeat" instruction closes it.
        """
        if self.timed_out:
            verdict = "STATUS: FAILED (tool timed out)"
        elif self.ok:
            verdict = "STATUS: OK"
        else:
            verdict = "STATUS: FAILED (exit %d)" % self.returncode

        body = (self.stdout or self.stderr).strip() or "(no output)"
        body = shorten_workspace_paths(body)
        body = re.sub(r"\s+", " ", body).strip()

        note = (
            "Do not call this tool again: answer with one final_answer naming the output."
            if self.ok and not self.timed_out
            else "Fix the arguments from the error above; do not repeat the identical call."
        )
        # The whole message stays within max_chars, so a long tool dump can
        # never inflate the model context. When the caller asks for a very small
        # budget, the fixed overhead (verdict + instruction) is dropped first so
        # the actual result still fits.
        overhead = len(verdict) + len(note) + 6
        if overhead > max_chars * 0.55:
            note = ""
            overhead = len(verdict) + 2
        budget = max(20, max_chars - overhead)
        if self.ok and not self.timed_out:
            budget = min(budget, 1000)
        if len(body) > budget:
            body = body[:budget] + " ...[truncated]"
        return ("%s\n%s\n[%s]" % (verdict, body, note)) if note else ("%s\n%s" % (verdict, body))


# --------------------------------------------------------------------------- #
# Session metadata
# --------------------------------------------------------------------------- #
class SessionMeta(BaseModel):
    session_id: str
    created_at: str
    updated_at: str
    title: str = "Untitled Chat"


class SessionContainer(BaseModel):
    metadata: SessionMeta
    messages: list[dict[str, Any]] = Field(default_factory=list)
