"""LLM client abstraction with pluggable providers (Ollama / LM Studio).

Layers
------
- **Transport** (`LLMTransport` protocol): wraps a specific inference backend
  behind two calls -- ``chat_json`` (request a JSON response) and ``chat_tools``
  (native function calling). Implementations: :class:`OllamaTransport` and
  :class:`LMStudioTransport` (OpenAI-compatible).
- **Strategies** (:class:`JSONStructuredLLMClient`,
  :class:`NativeToolLLMClient`): provider-agnostic decision logic built on a
  transport. This is what makes the agent loop unit-testable without a live model.
- **Auto** (:class:`AutoLLMClient`): picks native tool calling when enabled and
  falls back to structured JSON otherwise / when the model emits no tool calls.
"""

from __future__ import annotations

import json
import logging
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
    raise ValueError("Unbalanced JSON object in model output")


# --------------------------------------------------------------------------- #
# Transports (provider-specific API adapters)
# --------------------------------------------------------------------------- #
class LLMTransport(Protocol):
    """Provider adapter. Implementations normalize each backend's API."""

    name: str

    def chat_json(self, messages: list[dict[str, Any]], temperature: float) -> str:
        """Request a response formatted as JSON; return the text."""

    def chat_tools(
        self, messages: list[dict[str, Any]], tools: list[dict], temperature: float
    ) -> tuple[str, list[dict[str, Any]]]:
        """Native function calling. Returns (text, [{name, arguments: dict}]).
        ``text`` may be empty when the model only emits tool calls."""


def _coerce_tool_calls(raw_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize diverse tool-call shapes into [{name, arguments: dict}]."""
    out: list[dict[str, Any]] = []
    for tc in raw_calls or []:
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if name:
            out.append({"name": name, "arguments": args})
    return out


class OllamaTransport:
    """Adapter for a local Ollama server (native /api/chat)."""

    name = "ollama"

    def __init__(self, base_url: str, model: str) -> None:
        import ollama  # import locally so ollama is only required for this backend

        self._client = ollama.Client(host=base_url)
        self._model = model

    def chat_json(self, messages: list[dict[str, Any]], temperature: float) -> str:
        resp = self._client.chat(
            model=self._model,
            messages=messages,
            format="json",
            options={"temperature": temperature},
        )
        return resp["message"]["content"]

    def chat_tools(
        self, messages: list[dict[str, Any]], tools: list[dict], temperature: float
    ) -> tuple[str, list[dict[str, Any]]]:
        resp = self._client.chat(
            model=self._model,
            messages=messages,
            tools=tools,
            options={"temperature": temperature},
        )
        msg = resp["message"]
        return msg.get("content", "") or "", _coerce_tool_calls(msg.get("tool_calls") or [])


class LMStudioTransport:
    """Adapter for LM Studio / any OpenAI-compatible local server (/v1).

    JSON mode is adaptive: it requests ``response_format={"type":"json_object"}``
    but falls back to no ``response_format`` when the server rejects it (some LM
    Studio versions / model templates only accept ``json_schema`` or ``text``).
    The structured-JSON strategy's robust extraction + retries handle that case.
    """

    name = "lmstudio"

    def __init__(self, base_url: str, model: str) -> None:
        """Adapt to both the modern SDK (openai>=1.0) and the legacy 0.x module.

        Legacy 0.28 ships on plenty of conda environments; refusing to start the
        whole server because of it is worse than supporting it, so the JSON path
        works on either API. Native tool calling still needs openai>=1.0.
        """
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - user-install concern
            raise RuntimeError(
                "LM Studio support requires the 'openai' package: "
                "pip install -U \"openai>=1.0\""
            ) from exc

        self._model = model
        self._endpoint = base_url.rstrip("/") + "/v1"
        # openai>=1.0 exposes the `OpenAI` client class; 0.x is a flat module.
        if hasattr(openai, "OpenAI"):
            from openai import BadRequestError, OpenAI

            self._legacy = False
            self._BadRequestError = BadRequestError
            # LM Studio ignores the API key but the SDK requires a non-empty value.
            self._client = OpenAI(api_key="lm-studio", base_url=self._endpoint)
        else:  # pragma: no cover - depends on the installed SDK version
            self._legacy = True
            self._openai = openai
            self._BadRequestError = getattr(openai, "BadRequestError", Exception)
            openai.api_base = self._endpoint
            openai.api_key = "lm-studio"
            self._client = None

    def chat_json(self, messages: list[dict[str, Any]], temperature: float) -> str:
        if self._legacy:  # openai 0.x: flat module API, no response_format support
            resp = self._openai.ChatCompletion.create(
                model=self._model, messages=messages, temperature=temperature
            )
            return resp["choices"][0]["message"].get("content") or ""

        kwargs: dict[str, Any] = dict(
            model=self._model, messages=messages, temperature=temperature
        )
        try:
            kwargs["response_format"] = {"type": "json_object"}
            resp = self._client.chat.completions.create(**kwargs)
        except self._BadRequestError:
            logger.warning(
                "LM Studio rejected response_format=json_object; retrying without it"
            )
            kwargs.pop("response_format", None)
            resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def chat_tools(
        self, messages: list[dict[str, Any]], tools: list[dict], temperature: float
    ) -> tuple[str, list[dict[str, Any]]]:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            tools=tools,
        )
        msg = resp.choices[0].message
        raw_calls = [
            {"function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in (msg.tool_calls or [])
        ]
        return (msg.content or ""), _coerce_tool_calls(raw_calls)


def build_transport(settings: Settings) -> LLMTransport:
    """Instantiate the transport matching ``settings.provider``."""
    if settings.provider == "lmstudio":
        return LMStudioTransport(settings.lmstudio_base_url, settings.model)
    return OllamaTransport(settings.ollama_base_url, settings.model)


# --------------------------------------------------------------------------- #
# Live context block
# --------------------------------------------------------------------------- #
def input_files_note(settings, limit: int = 40) -> str:
    """A short "what is actually in workspace/input" block for the system prompt.

    Small local models otherwise invent filenames ("input.pdf"), get a sandbox
    error, guess again, and burn the entire step budget on failed calls.
    """
    try:
        from .harness import list_input_files

        files = list_input_files(settings)
    except Exception:  # noqa: BLE001 - context is best-effort
        return ""
    if not files:
        return (
            "\n\n# Available files in workspace/input\n"
            "(empty) There are no input files right now. Ask the user to add them "
            "before calling any tool that needs an input."
        )
    lines = [
        "\n\n# Available files in workspace/input",
        "Use these exact relative names in tool arguments. Never invent a filename.",
    ]
    for f in files[:limit]:
        lines.append("- %s (%d KB)" % (f["name"], max(1, f["size"] // 1024)))
    if len(files) > limit:
        lines.append("- ... and %d more" % (len(files) - limit))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Strategy interface
# --------------------------------------------------------------------------- #
class LLMClient(Protocol):
    def decide(
        self, messages: list[dict[str, Any]], max_steps_remaining: int
    ) -> AgentDecision: ...


def _normalize_decision(raw: dict[str, Any]) -> AgentDecision:
    return AgentDecision.model_validate(raw)


# --------------------------------------------------------------------------- #
# Structured-JSON strategy
# --------------------------------------------------------------------------- #
class JSONStructuredLLMClient:
    """Deterministic structured output with retry / self-correction."""

    def __init__(
        self,
        transport: LLMTransport,
        settings: Settings,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self._transport = transport
        self._settings = settings
        self._system_prompt = system_prompt

    def decide(
        self, messages: list[dict[str, Any]], max_steps_remaining: int
    ) -> AgentDecision:
        system = {
            "role": "system",
            "content": self._system_prompt + input_files_note(self._settings),
        }
        attempts = 0
        max_attempts = self._settings.json_retries + 1
        while attempts < max_attempts:
            attempt_messages = [system] + messages
            if attempts > 0:
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
                messages = attempt_messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"Your output was invalid: {exc}. Fix it."},
                ]
        return AgentDecision(
            need_ask_user=True,
            question="The assistant produced an invalid response. Please rephrase your request.",
        )

    def _generate(self, messages: list[dict[str, Any]]) -> str:
        return self._transport.chat_json(messages, self._settings.temperature)


# --------------------------------------------------------------------------- #
# Native tool-calling strategy
# --------------------------------------------------------------------------- #
class NativeToolLLMClient:
    """Uses the provider's native function-calling API when supported."""

    def __init__(
        self,
        transport: LLMTransport,
        settings: Settings,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self._transport = transport
        self._settings = settings
        self._system_prompt = system_prompt

    def decide(
        self, messages: list[dict[str, Any]], max_steps_remaining: int
    ) -> AgentDecision:
        tools = tool_schemas_for_llm()
        text, tool_calls = self._transport.chat_tools(
            [
                {
                    "role": "system",
                    "content": self._system_prompt + input_files_note(self._settings),
                }
            ] + messages,
            tools,
            self._settings.temperature,
        )

        if tool_calls:
            first = tool_calls[0]
            return AgentDecision(tool_name=first["name"], tool_args=first["arguments"])

        # Model did not emit tool calls: fall back to structured JSON reasoning.
        if text.strip():
            try:
                return _normalize_decision(extract_json_object(text))
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

    def __init__(self, transport: LLMTransport, settings: Settings):
        self._settings = settings
        self._json = JSONStructuredLLMClient(transport, settings)
        self._native = (
            NativeToolLLMClient(transport, settings) if settings.enable_native_tools else None
        )

    def decide(
        self, messages: list[dict[str, Any]], max_steps_remaining: int
    ) -> AgentDecision:
        if self._native is not None:
            decision = self._native.decide(messages, max_steps_remaining)
            if decision.is_tool_call:
                return decision
        return self._json.decide(messages, max_steps_remaining)
