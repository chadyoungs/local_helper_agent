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
import shutil
from pathlib import Path

import ollama
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .agent_loop import AgentRunner
from .llm_client import AutoLLMClient
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


def build_app(settings: Settings = DEFAULT_SETTINGS) -> FastAPI:
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
    llm_client = AutoLLMClient(ollama.Client(host=settings.ollama_base_url), settings)
    agent = AgentRunner(settings, llm_client, tool_runner)

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
        files = [
            {"name": p.name, "size": p.stat().st_size}
            for p in settings.input_root.iterdir()
            if p.is_file()
        ]
        files.sort(key=lambda f: f["name"].lower())
        return {"files": files}

    @app.post("/upload")
    async def api_upload(file: UploadFile = File(...)):
        # Reject dangerous names before they hit the filesystem.
        name = Path(file.filename or "").name  # strips any directory components
        if not name:
            raise HTTPException(status_code=400, detail="Empty filename")
        if name.startswith("."):
            raise HTTPException(status_code=400, detail="Hidden files are not allowed")
        dest = settings.input_root / name
        try:
            with dest.open("wb") as out:
                shutil.copyfileobj(file.file, out)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Upload failed: {exc}")
        return {"ok": True, "name": name, "path": str(dest)}

    # ------------------------- chat API ------------------------- #
    @app.post("/chat")
    async def api_chat(req: ChatRequest):
        container = sessions.load(req.session_id)
        if container is None:
            raise HTTPException(status_code=404, detail="Session not found")

        history: list[dict] = list(container.messages)  # snapshot for this turn

        async def stream():
            # We buffer events and flush them as NDJSON lines in order.
            from collections import deque
            buffer: deque = deque()
            emit = buffer.append

            error: str | None = None
            # Run the agent loop in a worker thread (Ollama client is blocking).
            try:
                await asyncio.to_thread(agent.run, history, req.user_message, emit)
            except Exception as exc:  # noqa: BLE001 - surface cleanly to client
                logger.exception("agent loop failed")
                error = str(exc)
            finally:
                sessions.save(req.session_id, history)

            if error is not None:
                yield json.dumps({
                    "event": "error",
                    "message": "Backend error while processing your request. "
                               "Check that the local Ollama service is running.",
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
