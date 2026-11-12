"""The modern agent loop.

Each turn follows a bounded ``plan -> tool -> observe -> conclude`` cycle:

1. Ask the LLM for a structured decision.
2. If it needs a clarification from the user -> emit ``ask_user`` and stop.
3. If it names a tool -> run it inside the harness sandbox, feed the sanitized
   result back as a ``tool`` message, and loop (subject to a step budget).
4. Otherwise -> the decision carries a ``final_answer``; that ends the turn.

This gives the agent genuine multi-step / tool-calling behaviour with a hard
budget so a misbehaving model cannot loop forever.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any, Dict, Optional

from .llm_client import LLMClient
from .schemas import AgentDecision, ToolResult, shorten_workspace_paths
from .settings import Settings
from .tools_runner import ToolRunner

logger = logging.getLogger("agent.loop")


class AgentRunner:
    def __init__(
        self,
        settings: Settings,
        llm_client: LLMClient,
        tool_runner: ToolRunner,
    ) -> None:
        self.settings = settings
        self.llm = llm_client
        self.tools = tool_runner

    def run(
        self,
        history: list[dict[str, Any]],
        user_message: str,
        emit: Callable[[dict[str, Any]], None],
        output_stem: Optional[str] = None,
    ) -> None:
        """Execute one user turn against an in-memory ``history`` list.

        ``emit`` receives NDJSON event dicts (chunk / tool_result / ask_user /
        final). The caller owns persistence of ``history``. ``output_stem`` is
        the per-turn ``<chat title>_<timestamp>`` folder name passed to the
        tool runner for ``workspace/output``.
        """
        history.append({"role": "user", "content": user_message})
        emit({"event": "user", "content": user_message})

        steps = 0
        budget = self.settings.max_agent_steps
        # Loop guards: remember what already succeeded / already failed so a
        # small model cannot burn the whole budget repeating itself.
        succeeded: Dict[str, ToolResult] = {}
        failed_once: Dict[str, ToolResult] = {}
        last_tool: Optional[str] = None
        tool_streak = 0
        last_result: Optional[ToolResult] = None

        while steps < budget:
            decision = self.llm.decide(history, budget - steps)
            logger.debug("decision | question=%s tool=%s final=%s",
                         decision.need_ask_user, decision.tool_name, decision.is_final)

            # 1. Clarification request -> stop the turn, let the user answer next message.
            if decision.is_question:
                record = json.dumps(decision.model_dump(), ensure_ascii=False)
                history.append({"role": "assistant", "content": record})
                emit({"event": "ask_user", "question": decision.question})
                return

            # 2. Tool call -> run it and feed the result back.
            if decision.is_tool_call:
                signature = _call_signature(decision.tool_name, decision.tool_args)

                # 2a. The exact same call already succeeded: stop instead of looping.
                if signature in succeeded:
                    answer = _answer_from_result(decision.tool_name, succeeded[signature])
                    history.append({"role": "assistant", "content": answer})
                    emit({"event": "chunk", "content": answer})
                    emit({"event": "final", "content": answer})
                    return

                # 2b. The exact same call failed twice already: ask, don't guess again.
                if signature in failed_once:
                    question = _retry_question(decision.tool_name, failed_once[signature])
                    record = json.dumps(decision.model_dump(), ensure_ascii=False)
                    history.append({"role": "assistant", "content": record})
                    emit({"event": "ask_user", "question": question})
                    return

                record = json.dumps(decision.model_dump(), ensure_ascii=False)
                history.append({"role": "assistant", "content": record})
                emit({"event": "tool_call", "tool": decision.tool_name, "args": decision.tool_args})

                result = self.tools.run(decision.tool_name, decision.tool_args, output_stem)
                emit({"event": "tool_result", "tool": decision.tool_name, "payload": result.model_dump()})

                summary = result.summary(max_chars=self.settings.tool_output_max_chars)
                history.append({"role": "tool", "name": decision.tool_name, "content": summary})

                tool_streak = tool_streak + 1 if decision.tool_name == last_tool else 1
                last_tool = decision.tool_name
                last_result = result
                steps += 1

                if result.ok:
                    succeeded[signature] = result
                    # 2c. A long churn of the same tool (4+ attempts, each with
                    # different arguments) is wasteful; report the last success.
                    if tool_streak >= 4:
                        answer = _answer_from_result(decision.tool_name, result)
                        history.append({"role": "assistant", "content": answer})
                        emit({"event": "chunk", "content": answer})
                        emit({"event": "final", "content": answer})
                        return
                else:
                    failed_once[signature] = result
                continue

            # 3. Final answer -> done.
            if decision.is_final:
                answer = decision.final_answer
                history.append({"role": "assistant", "content": answer})
                emit({"event": "chunk", "content": answer})
                emit({"event": "final", "content": answer})
                return

            # 4. Degenerate decision (all fields empty) -> force a clean end.
            logger.warning("empty decision at step %s", steps)
            fallback = (
                "I could not determine an action to take. "
                "Please clarify what you would like me to do."
            )
            history.append({"role": "assistant", "content": fallback})
            emit({"event": "chunk", "content": fallback})
            emit({"event": "final", "content": fallback})
            return

        # Budget exhausted without a final answer.
        if last_result is not None and last_result.ok:
            msg = _answer_from_result(last_tool or "tool", last_result)
        else:
            detail = ""
            if last_result is not None:
                detail = " Last error: " + last_result.summary(max_chars=300)
            msg = (
                f"I stopped after {budget} steps to avoid repeating the same call."
                + detail
                + " If you still need this done, reply with the exact input filename(s)."
            )
        history.append({"role": "assistant", "content": msg})
        emit({"event": "chunk", "content": msg})
        emit({"event": "final", "content": msg})


def _call_signature(tool_name: str, tool_args: Dict[str, Any]) -> str:
    """Stable key for "the same tool with the same arguments"."""
    return tool_name + "|" + json.dumps(tool_args, sort_keys=True, ensure_ascii=False)


def _answer_from_result(tool_name: str, result: ToolResult) -> str:
    """Human-facing one-liner describing a successful tool run."""
    data: Dict[str, Any] = {}
    raw = (result.stdout or "").strip()
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
        except ValueError:
            data = {}
    output = data.get("output") or data.get("output_file") or ""
    if output:
        return "Done: %s wrote %s" % (tool_name, shorten_workspace_paths(str(output)))
    return "Done: %s completed (status OK)." % tool_name


def _retry_question(tool_name: str, result: ToolResult) -> str:
    """Ask the user for the missing information after a repeated failure."""
    reason = result.summary(max_chars=300)
    return (
        "`%s` failed and the identical call was repeated, so I stopped instead of "
        "retrying blindly.\n%s\nCan you give me the exact filename or the value "
        "it needs?" % (tool_name, reason)
    )
