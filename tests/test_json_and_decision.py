"""Tests for robust JSON extraction and structured-decision parsing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.llm_client import JSONStructuredLLMClient, extract_json_object
from agent.schemas import AgentDecision
from agent.settings import Settings


def test_extract_pure_json():
    assert extract_json_object('{"tool_name":"pdf_merge"}')["tool_name"] == "pdf_merge"


def test_extract_from_code_fence():
    text = '```json\n{"tool_name":"pdf2img"}\n```'
    assert extract_json_object(text)["tool_name"] == "pdf2img"


def test_extract_with_surrounding_prose():
    text = 'Sure! Here you go: {"need_ask_user":false,"final_answer":"done"} thanks!'
    d = extract_json_object(text)
    assert d["final_answer"] == "done"


def test_extract_nested_braces_in_strings():
    text = '{"question":"what is {this}?", "final_answer":"ok"}'
    d = extract_json_object(text)
    assert d["question"] == "what is {this}?"


def test_extract_no_json_raises():
    with pytest.raises(ValueError):
        extract_json_object("no json here")


def test_extract_unbalanced_raises():
    with pytest.raises(ValueError):
        extract_json_object('{"a": 1')


class _FakeOllama:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def chat(self, **kwargs):
        raw = self.outputs.pop(0)
        return {"message": {"content": raw}}


def _client(outputs, retries=1):
    settings = Settings(json_retries=retries)
    return JSONStructuredLLMClient(_FakeOllama(outputs), settings)


def test_structured_client_parses_valid():
    client = _client(['{"final_answer":"hi"}'])
    d = client.decide([{"role": "user", "content": "hi"}], 5)
    assert d.final_answer == "hi"


def test_structured_client_recovers_on_retry():
    # first output invalid, second valid
    client = _client(['not json at all', '{"tool_name":"pdf_merge","tool_args":{"inputs":"a.pdf"}}'], retries=1)
    d = client.decide([{"role": "user", "content": "merge"}], 5)
    assert d.is_tool_call
    assert d.tool_name == "pdf_merge"


def test_structured_client_asks_user_after_exhaustion():
    client = _client(["bad1", "bad2", "bad3"], retries=1)  # retries=1 -> max 2 attempts
    d = client.decide([{"role": "user", "content": "x"}], 5)
    assert d.is_question
    assert d.question


def test_tool_result_summary_truncates():
    from agent.schemas import ToolResult
    res = ToolResult(ok=True, stdout="x" * 5000)
    s = res.summary(max_chars=100)
    assert len(s) <= 100 + len("\n...[truncated]")
    assert "truncated" in s
