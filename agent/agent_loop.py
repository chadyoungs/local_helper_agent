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
from typing import Any

from .llm_client import LLMClient
from .schemas import AgentDecision
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
    ) -> None:
        """Execute one user turn against an in-memory ``history`` list.

        ``emit`` receives NDJSON event dicts (chunk / tool_result / ask_user /
        final). The caller owns persistence of ``history``.
        """
        history.append({"role": "user", "content": user_message})
        emit({"event": "user", "content": user_message})

        steps = 0
        budget = self.settings.max_agent_steps

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
                record = json.dumps(decision.model_dump(), ensure_ascii=False)
                history.append({"role": "assistant", "content": record})
                emit({"event": "tool_call", "tool": decision.tool_name, "args": decision.tool_args})

                result = self.tools.run(decision.tool_name, decision.tool_args)
                emit({"event": "tool_result", "tool": decision.tool_name, "payload": result.model_dump()})

                summary = result.summary(max_chars=self.settings.tool_output_max_chars)
                history.append({"role": "tool", "name": decision.tool_name, "content": summary})
                steps += 1
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
        msg = (
            f"I exceeded the step budget ({budget}) while trying to complete this. "
            "Please narrow the request or check the tool results above."
        )
        history.append({"role": "assistant", "content": msg})
        emit({"event": "chunk", "content": msg})
        emit({"event": "final", "content": msg})
