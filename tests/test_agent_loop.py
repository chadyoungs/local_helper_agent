"""Tests for the modern agent loop: multi-step tool calls, result feedback,
ask_user, final answers, and the step budget."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.agent_loop import AgentRunner
from agent.llm_client import AgentDecision as _AD  # noqa
from agent.schemas import AgentDecision, ToolResult
from agent.settings import Settings
from agent.tool_registry import TOOL_REGISTRY
from agent.tools_runner import ToolRunner


class _FakeLLM:
    """Deterministic scripted LLM: pops decisions from a queue."""

    def __init__(self, decisions):
        self.decisions = list(decisions)

    def decide(self, messages, max_steps_remaining):
        if not self.decisions:
            return AgentDecision(final_answer="out of script")
        return self.decisions.pop(0)


class _FakeTools:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def run(self, tool_name, tool_args):
        self.calls.append((tool_name, tool_args))
        if self.results:
            return self.results.pop(0)
        return ToolResult(ok=True, stdout="ok")


def _settings() -> Settings:
    s = Settings(workspace_root=Path(__file__).resolve().parent / "_test_ws2")
    s.ensure_dirs()
    return s


def _run(history, user_msg, llm, tools):
    runner = AgentRunner(_settings(), llm, tools)
    events = []
    runner.run(history, user_msg, emit=events.append)
    return history, events


# --------------------------------------------------------------------------- #
def test_single_tool_call_then_final_answer():
    history = []
    llm = _FakeLLM([
        AgentDecision(tool_name="pdf_merge", tool_args={"inputs": "a.pdf b.pdf", "output": "m.pdf"}),
        AgentDecision(final_answer="Merged successfully."),
    ])
    tools = _FakeTools([ToolResult(ok=True, stdout="ok")])
    history, events = _run(history, "merge these", llm, tools)

    assert tools.calls == [("pdf_merge", {"inputs": "a.pdf b.pdf", "output": "m.pdf"})]
    # tool result fed back as a 'tool' role message
    assert any(m["role"] == "tool" for m in history)
    # final assistant answer recorded
    assert any(m["role"] == "assistant" and m["content"] == "Merged successfully." for m in history)
    assert any(e["event"] == "final" for e in events)
    assert any(e["event"] == "tool_result" for e in events)


def test_multiple_tool_calls_in_one_turn():
    history = []
    llm = _FakeLLM([
        AgentDecision(tool_name="pdf2img", tool_args={"inputs": ["a.pdf"], "output": "imgs"}),
        AgentDecision(tool_name="id_photo_bg_change", tool_args={"input": "x.jpg", "output": "o.jpg", "target_color": "#FF0000"}),
        AgentDecision(final_answer="Done both."),
    ])
    tools = _FakeTools([ToolResult(ok=True, stdout="a"), ToolResult(ok=True, stdout="b")])
    history, _ = _run(history, "do both", llm, tools)

    assert len(tools.calls) == 2
    assert sum(1 for m in history if m["role"] == "tool") == 2


def test_ask_user_stops_turn():
    history = []
    llm = _FakeLLM([AgentDecision(need_ask_user=True, question="Need the PDF password?")])
    tools = _FakeTools([])
    history, events = _run(history, "open pdf", llm, tools)

    assert tools.calls == []
    assert any(e["event"] == "ask_user" for e in events)
    # no tool messages recorded
    assert not any(m["role"] == "tool" for m in history)


def test_step_budget_enforced():
    history = []
    # model keeps calling tools, never concludes
    llm = _FakeLLM([
        AgentDecision(tool_name="pdf_merge", tool_args={"inputs": "a.pdf", "output": "o.pdf"})
        for _ in range(99)
    ])
    tools = _FakeTools([ToolResult(ok=True, stdout="ok") for _ in range(99)])
    history, events = _run(history, "loop", llm, tools)

    max_steps = _settings().max_agent_steps
    assert len(tools.calls) == max_steps
    # loop terminates with a budget message
    final_event = next(e for e in reversed(events) if e["event"] == "final")
    assert "budget" in final_event["content"]


def test_final_answer_without_tool():
    history = []
    llm = _FakeLLM([AgentDecision(final_answer="Hello!")])
    tools = _FakeTools([])
    history, events = _run(history, "hi", llm, tools)
    assert tools.calls == []
    assert any(e["event"] == "final" and e["content"] == "Hello!" for e in events)


def test_empty_decision_falls_back():
    history = []
    llm = _FakeLLM([AgentDecision()])
    tools = _FakeTools([])
    history, events = _run(history, "x", llm, tools)
    final = next(e for e in events if e["event"] == "final")
    assert final["content"]  # non-empty fallback
