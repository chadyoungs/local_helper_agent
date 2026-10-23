"""Pydantic models shared across the agent.

These are the single source of truth for (a) how the model's structured output
is validated and (b) how tools are described to the LLM.
"""

from __future__ import annotations

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


class ToolResult(BaseModel):
    """Unified, sanitized result of running an external tool."""

    ok: bool = False
    stdout: str = ""
    stderr: str = ""
    returncode: int = -1
    timed_out: bool = False

    def summary(self, max_chars: int = 2000) -> str:
        """Compact text fed back to the model so history does not blow up."""
        parts: list[str] = []
        if self.stdout.strip():
            parts.append(f"stdout: {self.stdout.strip()}")
        if self.stderr.strip():
            parts.append(f"stderr: {self.stderr.strip()}")
        body = "\n".join(parts)
        if self.timed_out:
            body += "\n[tool timed out]"
        if not body:
            body = "(no output)"
        if len(body) > max_chars:
            body = body[:max_chars] + "\n...[truncated]"
        return body


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
