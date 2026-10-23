"""Integration tests for the FastAPI app using TestClient and a fake LLM.

No live Ollama is required: we inject a scripted LLM into the built app.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.api import build_app
from agent.schemas import AgentDecision, ToolResult
from agent.settings import Settings


class _FakeLLM:
    def __init__(self, decisions):
        self.decisions = list(decisions)

    def decide(self, messages, max_steps_remaining):
        if not self.decisions:
            return AgentDecision(final_answer="done")
        return self.decisions.pop(0)


@pytest.fixture()
def client(tmp_path):
    settings = Settings(workspace_root=tmp_path / "ws", sessions_dir=tmp_path / "sessions")
    app = build_app(settings)
    # inject fake LLM and fake tool runner
    app.state.agent.llm = _FakeLLM([
        AgentDecision(tool_name="pdf_merge", tool_args={"inputs": "a.pdf", "output": "m.pdf"}),
        AgentDecision(final_answer="Merged."),
    ])
    app.state.agent.tools = _FakeTools([ToolResult(ok=True, stdout="ok")])
    return TestClient(app)


class _FakeTools:
    def __init__(self, results):
        self.results = list(results)

    def run(self, tool_name, tool_args):
        return self.results.pop(0)


def test_new_session_and_list(client):
    r = client.post("/new_session")
    assert r.status_code == 200
    sid = r.json()["session_id"]
    assert sid

    r = client.get("/sessions")
    assert r.status_code == 200
    assert any(s["session_id"] == sid for s in r.json()["sessions"])


def test_get_rename_delete_session(client):
    sid = client.post("/new_session").json()["session_id"]

    r = client.patch(f"/session/{sid}", json={"title": "My Chat"})
    assert r.status_code == 200

    r = client.get(f"/session/{sid}")
    assert r.json()["metadata"]["title"] == "My Chat"

    assert client.delete(f"/session/{sid}").json()["ok"] is True
    assert client.get(f"/session/{sid}").status_code == 404


def test_upload_and_list_files(client, tmp_path):
    # note: build_app used tmp_path/ws as workspace
    r = client.post("/upload", files={"file": ("report.pdf", b"%PDF-1.4 fake", "application/pdf")})
    assert r.status_code == 200
    assert r.json()["name"] == "report.pdf"

    r = client.get("/files")
    assert any(f["name"] == "report.pdf" for f in r.json()["files"])


def test_upload_rejects_path_traversal_name(client):
    r = client.post("/upload", files={"file": ("../evil.pdf", b"x", "application/pdf")})
    # FastAPI/starlette strips the basename, so ../evil.pdf becomes evil.pdf
    assert r.status_code == 200
    assert r.json()["name"] == "evil.pdf"


def test_chat_streams_ndjson_events(client):
    sid = client.post("/new_session").json()["session_id"]
    r = client.post("/chat", json={"session_id": sid, "user_message": "merge these"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    lines = r.text.strip().split("\n")
    events = [json.loads(l) for l in lines]
    assert any(e["event"] == "user" for e in events)
    assert any(e["event"] == "tool_result" for e in events)
    assert any(e["event"] == "final" for e in events)
    assert events[-1]["event"] == "done"


def test_chat_404_for_unknown_session(client):
    r = client.post("/chat", json={"session_id": "nope", "user_message": "hi"})
    assert r.status_code == 404
