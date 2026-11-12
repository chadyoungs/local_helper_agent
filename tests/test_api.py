"""Integration tests for the FastAPI app using TestClient and a fake LLM.

No live model and no provider package is required: the transport and the
LLM/tool runner are injected into the built app.
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


class _DummyTransport:
    """Stands in for Ollama/LM Studio so build_app() needs neither package."""

    name = "dummy"

    def chat_json(self, messages, temperature):
        return "{}"

    def chat_tools(self, messages, tools, temperature):
        return "", []


class _FakeLLM:
    def __init__(self, decisions):
        self.decisions = list(decisions)

    def decide(self, messages, max_steps_remaining):
        if not self.decisions:
            return AgentDecision(final_answer="done")
        return self.decisions.pop(0)


class _FakeTools:
    def __init__(self, results):
        self.results = list(results)

    def run(self, tool_name, tool_args, output_stem=None):
        return self.results.pop(0)


@pytest.fixture()
def client(tmp_path):
    settings = Settings(
        workspace_root=tmp_path / "ws",
        sessions_dir=tmp_path / "sessions",
    )
    app = build_app(settings, transport=_DummyTransport())
    app.state.agent.llm = _FakeLLM([
        AgentDecision(tool_name="pdf_merge", tool_args={"inputs": "a.pdf", "output": "m.pdf"}),
        AgentDecision(final_answer="Merged."),
    ])
    app.state.agent.tools = _FakeTools([ToolResult(ok=True, stdout='{"ok": true}')])
    return TestClient(app)


@pytest.fixture()
def ask_client(tmp_path):
    settings = Settings(
        workspace_root=tmp_path / "ws2",
        sessions_dir=tmp_path / "sessions2",
    )
    app = build_app(settings, transport=_DummyTransport())
    app.state.agent.llm = _FakeLLM([
        AgentDecision(need_ask_user=True, question="Which lecture hall do you mean?"),
    ])
    app.state.agent.tools = _FakeTools([])
    return TestClient(app)


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


def test_files_lists_every_file_including_subfolders(client, tmp_path):
    """The panel must show ALL files under workspace/input, not just the top level."""
    input_root = tmp_path / "ws" / "input"
    (input_root / "2024" / "scans").mkdir(parents=True, exist_ok=True)
    (input_root / "top.pdf").write_bytes(b"%PDF-1.4 fake")
    (input_root / "2024" / "scan.jpg").write_bytes(b"fake-jpeg")
    (input_root / "2024" / "scans" / "deep.png").write_bytes(b"fake-png")
    (input_root / ".gitkeep").write_text("")

    r = client.get("/files")
    assert r.status_code == 200
    body = r.json()
    names = [f["name"] for f in body["files"]]
    assert names == ["2024/scans/deep.png", "2024/scan.jpg", "top.pdf"] or set(names) == {
        "2024/scans/deep.png",
        "2024/scan.jpg",
        "top.pdf",
    }
    assert body["count"] == 3
    assert ".gitkeep" not in names
    # `dir` tells the frontend how to group the list
    by_name = {f["name"]: f for f in body["files"]}
    assert by_name["2024/scan.jpg"]["dir"] == "2024"
    assert by_name["2024/scans/deep.png"]["dir"] == "2024/scans"
    assert by_name["top.pdf"]["dir"] == ""


def test_chat_streams_ndjson_events(client):
    sid = client.post("/new_session").json()["session_id"]
    r = client.post("/chat", json={"session_id": sid, "user_message": "merge these"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    lines = r.text.strip().split("\n")
    events = [json.loads(line) for line in lines]
    assert any(e["event"] == "user" for e in events)
    assert any(e["event"] == "tool_result" for e in events)
    assert any(e["event"] == "final" for e in events)
    assert events[-1]["event"] == "done"


def test_chat_emits_ask_user_event_inline(ask_client):
    """Clarifications travel as an `ask_user` event (rendered in the chat stream)."""
    sid = ask_client.post("/new_session").json()["session_id"]
    r = ask_client.post("/chat", json={"session_id": sid, "user_message": "cut my signature"})
    events = [json.loads(line) for line in r.text.strip().split("\n") if line.strip()]
    ask = [e for e in events if e["event"] == "ask_user"]
    assert ask and "lecture hall" in ask[0]["question"]
    assert events[-1]["event"] == "done"


def test_chat_404_for_unknown_session(client):
    r = client.post("/chat", json={"session_id": "nope", "user_message": "hi"})
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# Regression guards for the "failed to create session" failure mode
# --------------------------------------------------------------------------- #
def test_sessions_list_survives_corrupt_file(tmp_path):
    """One truncated/empty session file must not break the whole session list."""
    settings = Settings(workspace_root=tmp_path / "ws3", sessions_dir=tmp_path / "sessions3")
    client = TestClient(build_app(settings, transport=_DummyTransport()))

    sid = client.post("/new_session").json()["session_id"]
    (tmp_path / "sessions3" / "00000000-broken.json").write_bytes(b"")
    (tmp_path / "sessions3" / "11111111-broken.json").write_text("{not json", encoding="utf-8")

    r = client.get("/sessions")
    assert r.status_code == 200
    assert any(s["session_id"] == sid for s in r.json()["sessions"])


def test_app_starts_when_provider_sdk_is_unavailable(tmp_path, monkeypatch):
    """A broken LLM backend must not stop sessions / the file panel from working."""
    import agent.api as api_module

    def _boom(_settings):
        raise RuntimeError("LM Studio support requires the 'openai' package")

    monkeypatch.setattr(api_module, "build_transport", _boom)
    settings = Settings(workspace_root=tmp_path / "ws4", sessions_dir=tmp_path / "sessions4")
    client = TestClient(build_app(settings))

    assert client.post("/new_session").status_code == 200
    assert client.get("/files").status_code == 200

    sid = client.post("/new_session").json()["session_id"]
    r = client.post("/chat", json={"session_id": sid, "user_message": "hi"})
    assert r.status_code == 200
    events = [json.loads(line) for line in r.text.strip().splitlines()]
    assert any(e["event"] == "error" for e in events)
    assert any("openai" in (e.get("detail") or "") for e in events)


def test_tool_subprocess_runs_with_backend_interpreter(tmp_path, monkeypatch):
    """Tools must use the backend's interpreter, not a bare 'python3' on PATH."""
    import agent.tools_runner as runner_module

    from agent.schemas import ToolSpec
    from agent.tools_runner import ToolRunner

    settings = Settings(workspace_root=tmp_path / "ws5", sessions_dir=tmp_path / "sessions5")
    settings.ensure_dirs()
    assert settings.tool_python == sys.executable

    captured = {}

    class _Proc:
        returncode = 0
        stdout = '{"ok": true}'
        stderr = ""

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _Proc()

    monkeypatch.setattr(runner_module.subprocess, "run", _fake_run)

    spec = ToolSpec(name="t", description="d", script_path="/tmp/tool_script.py", parameters=[])
    runner = ToolRunner(settings)
    runner._by_name["t"] = spec  # register the stand-in spec for this test
    runner.run("t", {})

    assert captured["cmd"][0] == sys.executable
    assert captured["cmd"][1] == "/tmp/tool_script.py"
