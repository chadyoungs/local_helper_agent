"""FastAPI application: HTTP + NDJSON streaming chat, session management,
file upload and input-file listing.

Frontend and backend are decoupled: the chat endpoint yields an NDJSON stream
of events (user / tool_call / tool_result / ask_user / chunk / final) which the
vanilla-JS frontend renders.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .agent_loop import AgentRunner
from .harness import build_output_dir_name, list_input_files
from .llm_client import AutoLLMClient, LLMTransport, build_transport
from .schemas import ToolSpec
from .session_store import SessionStore
from .settings import DEFAULT_SETTINGS, Settings
from .tool_registry import SYSTEM_PROMPT, TOOL_REGISTRY
from .tools_runner import ToolRunner

logger = logging.getLogger("agent.api")


class ChatRequest(BaseModel):
    session_id: str
    user_message: str


class RenameRequest(BaseModel):
    title: str


def build_app(
    settings: Settings = DEFAULT_SETTINGS,
    transport: Optional[LLMTransport] = None,
) -> FastAPI:
    """Build the FastAPI app.

    ``transport`` lets tests (or an embedding caller) inject an LLM transport
    instead of constructing the Ollama / LM Studio adapter, so the app can be
    exercised without the provider package installed or a server running.
    """
    settings.ensure_dirs()
    app = FastAPI(title="Local Helper Agent", version="2.0.0")

    # --- CORS: explicit local origins only; credentials disabled ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    sessions = SessionStore(settings)
    tool_runner = ToolRunner(settings)

    # The LLM client is built on FIRST USE, not at import/start-up. A missing or
    # outdated provider SDK (e.g. openai 0.x with LLM_PROVIDER=lmstudio) used to
    # abort the whole server, which surfaced in the UI as a useless
    # "failed to create session". Sessions and /files now work regardless, and
    # /chat reports the real reason inside its NDJSON error event.
    _cache: dict = {"client": None, "error": None}

    def _resolve_client():
        if _cache["client"] is not None:
            return _cache["client"]
        if _cache["error"] is not None:
            raise _cache["error"]
        try:
            _cache["client"] = AutoLLMClient(transport or build_transport(settings), settings)
        except Exception as exc:  # noqa: BLE001 - replayed per request
            _cache["error"] = exc
            raise
        return _cache["client"]

    class _LazyLLMClient:
        """Proxy that resolves the real client on the first decision."""

        def decide(self, messages, max_steps_remaining):
            return _resolve_client().decide(messages, max_steps_remaining)

    agent = AgentRunner(settings, _LazyLLMClient(), tool_runner)

    # Expose internals so tests can inject a fake LLM / runner.
    app.state.sessions = sessions
    app.state.tool_runner = tool_runner
    app.state.agent = agent

    # ------------------------- session APIs ------------------------- #
    @app.post("/new_session")
    def api_new_session():
        sid, _ = sessions.create()
        return {"session_id": sid}

    @app.get("/sessions")
    def api_list_sessions():
        return {"sessions": [m.model_dump() for m in sessions.list_meta()]}

    @app.get("/session/{session_id}")
    def api_get_session(session_id: str):
        container = sessions.load(session_id)
        if container is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return {
            "session_id": session_id,
            "metadata": container.metadata.model_dump(),
            "messages": container.messages,
        }

    @app.delete("/session/{session_id}")
    def api_del_session(session_id: str):
        sessions.delete(session_id)
        return {"ok": True}

    @app.patch("/session/{session_id}")
    def api_rename_session(session_id: str, req: RenameRequest):
        if not sessions.rename(session_id, req.title.strip() or "Untitled Chat"):
            raise HTTPException(status_code=404, detail="Session not found")
        return {"ok": True}

    # ------------------------- file APIs ------------------------- #
    @app.get("/files")
    def api_list_files():
        """List EVERY file under workspace/input, sub-folders included.

        ``name`` is the relative path (``2024/scan.jpg``), which is exactly the
        logical filename the harness accepts in tool arguments.
        """
        files = list_input_files(settings)
        return {
            "files": files,
            "count": len(files),
            "input_root": str(settings.input_root),
        }

    # ------------------------- chat API ------------------------- #
    @app.post("/chat")
    async def api_chat(req: ChatRequest):
        container = sessions.load(req.session_id)
        if container is None:
            raise HTTPException(status_code=404, detail="Session not found")

        history: list[dict] = list(container.messages)  # snapshot for this turn
        # Per-turn output folder under workspace/output, named "<chat title>_<time>".
        output_stem = build_output_dir_name(container.metadata.title)

        async def stream():
            # We buffer events and flush them as NDJSON lines in order.
            from collections import deque
            buffer: deque = deque()
            emit = buffer.append

            error: Optional[str] = None
            # Run the agent loop in a worker thread (the LLM client is blocking).
            try:
                await asyncio.to_thread(agent.run, history, req.user_message, emit, output_stem)
            except Exception as exc:  # noqa: BLE001 - surface cleanly to client
                logger.exception("agent loop failed")
                error = str(exc)
            finally:
                sessions.save(req.session_id, history)

            if error is not None:
                backend_hint = (
                    "Check that Ollama is running (ollama serve)."
                    if settings.provider == "ollama"
                    else "Check that LM Studio's local server is running on %s."
                    % settings.lmstudio_base_url
                )
                yield json.dumps({
                    "event": "error",
                    "message": "Backend error while processing your request. " + backend_hint,
                    "detail": error,
                }, ensure_ascii=False) + "\n"
            else:
                for event in buffer:
                    yield json.dumps(event, ensure_ascii=False) + "\n"
            yield json.dumps({"event": "done"}) + "\n"

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    return app


# --------------------------------------------------------------------------- #
# Dependency wiring helpers (used by tests too)
# --------------------------------------------------------------------------- #
def build_tool_runner(settings: Settings) -> ToolRunner:
    return ToolRunner(settings)


def registry_specs() -> list[ToolSpec]:
    return TOOL_REGISTRY
