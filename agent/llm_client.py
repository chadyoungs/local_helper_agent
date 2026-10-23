"""LLM client abstraction.

Two strategies are provided behind one interface so the agent loop is
transport-agnostic and unit-testable without a live model:

- :class:`NativeToolLLMClient`  -- uses Ollama's native function-calling API.
- :class:`JSONStructuredLLMClient` -- uses ``format="json"`` plus robust JSON
  extraction, schema validation, and bounded self-correction retries (the
  deterministic "local JSON agent + checker" pattern).

:class:`AutoLLMClient` picks native tools when enabled and falls back to the
structured-JSON strategy otherwise / when the model emits no tool calls.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol

from pydantic import ValidationError

from .schemas import AgentDecision
from .settings import Settings
from .tool_registry import SYSTEM_PROMPT, tool_schemas_for_llm

logger = logging.getLogger("agent.llm")


# --------------------------------------------------------------------------- #
# JSON robustness helpers
# --------------------------------------------------------------------------- #
def extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort extraction of a JSON object from an LLM response.

    Handles markdown code fences, leading prose, and trailing commentary by
    scanning for the outermost balanced ``{ ... }`` block.
    """
    stripped = text.strip()
    # Drop ```json ... ``` fences if present.
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1]) if len(lines) >= 2 else stripped
        stripped = stripped.strip()

    start = stripped.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(stripped)):
        ch = stripped[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : i + 1]
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
                raise ValueError("Extracted JSON is not an object")
        # ignore everything outside the top-level object
    raise ValueError("Unbalanced JSON object in model output")


# --------------------------------------------------------------------------- #
# Strategy interface
# --------------------------------------------------------------------------- #
class LLMClient(Protocol):
    def decide(
        self, messages: list[dict[str, Any]], max_steps_remaining: int
    ) -> AgentDecision: ...


def _normalize_decision(raw: dict[str, Any]) -> AgentDecision:
    """Coerce a raw dict into an AgentDecision, tolerating minor shape issues."""
    return AgentDecision.model_validate(raw)


# --------------------------------------------------------------------------- #
# Structured-JSON strategy
# --------------------------------------------------------------------------- #
class JSONStructuredLLMClient:
    """Deterministic structured output with retry / self-correction."""

    def __init__(self, client: Any, settings: Settings, system_prompt: str = SYSTEM_PROMPT):
        self._client = client
        self._settings = settings
        self._system_prompt = system_prompt

    def decide(
        self, messages: list[dict[str, Any]], max_steps_remaining: int
    ) -> AgentDecision:
        system = {"role": "system", "content": self._system_prompt}
        attempts = 0
        max_attempts = self._settings.json_retries + 1
        while attempts < max_attempts:
            attempt_messages = [system] + messages
            if attempts > 0:
                # Self-correction hint appended to the last attempt.
                attempt_messages = attempt_messages + [
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was invalid. Fix the error and "
                            "respond with ONLY a valid JSON object matching the schema."
                        ),
                    }
                ]
            raw = self._generate(attempt_messages)
            attempts += 1
            try:
                return _normalize_decision(extract_json_object(raw))
            except (ValidationError, ValueError) as exc:
                logger.warning("decision parse failed (attempt %s): %s", attempts, exc)
                # Remember the bad output for the next attempt's hint.
                messages = attempt_messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"Your output was invalid: {exc}. Fix it."},
                ]
        # Last resort: a safe, empty decision asking the user.
        return AgentDecision(
            need_ask_user=True,
            question="The assistant produced an invalid response. Please rephrase your request.",
        )

    def _generate(self, messages: list[dict[str, Any]]) -> str:
        response = self._client.chat(
            model=self._settings.model,
            messages=messages,
            format="json",
            options={"temperature": self._settings.temperature},
        )
        return response["message"]["content"]


# --------------------------------------------------------------------------- #
# Native tool-calling strategy
# --------------------------------------------------------------------------- #
class NativeToolLLMClient:
    """Uses Ollama's native function-calling API when the model supports it."""

    def __init__(self, client: Any, settings: Settings, system_prompt: str = SYSTEM_PROMPT):
        self._client = client
        self._settings = settings
        self._system_prompt = system_prompt

    def decide(
        self, messages: list[dict[str, Any]], max_steps_remaining: int
    ) -> AgentDecision:
        tools = tool_schemas_for_llm()
        response = self._client.chat(
            model=self._settings.model,
            messages=[{"role": "system", "content": self._system_prompt}] + messages,
            tools=tools,
            options={"temperature": self._settings.temperature},
        )
        msg = response["message"]
        tool_calls = msg.get("tool_calls") or []

        # Native tool call path.
        if tool_calls:
            first = tool_calls[0]
            fn = first.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if name:
                return AgentDecision(tool_name=name, tool_args=args)

        # Model did not emit tool calls: fall back to structured JSON reasoning.
        content = msg.get("content", "")
        if content.strip():
            try:
                return _normalize_decision(extract_json_object(content))
            except (ValidationError, ValueError) as exc:
                logger.warning("native fallback parse failed: %s", exc)
        return AgentDecision(
            need_ask_user=True,
            question="The assistant could not determine the next action. Please rephrase.",
        )


# --------------------------------------------------------------------------- #
# Auto strategy
# --------------------------------------------------------------------------- #
class AutoLLMClient:
    """Choose native tool-calling when enabled; otherwise structured JSON."""

    def __init__(self, client: Any, settings: Settings):
        self._settings = settings
        self._json = JSONStructuredLLMClient(client, settings)
        self._native = NativeToolLLMClient(client, settings) if settings.enable_native_tools else None

    def decide(
        self, messages: list[dict[str, Any]], max_steps_remaining: int
    ) -> AgentDecision:
        if self._native is not None:
            decision = self._native.decide(messages, max_steps_remaining)
            if decision.is_tool_call:
                return decision
            # otherwise fall through to structured JSON for reasoning
        return self._json.decide(messages, max_steps_remaining)
