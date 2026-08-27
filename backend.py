import asyncio
import json
import logging

import ollama
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent_core import (create_new_session, get_session_path,
                        list_sessions_with_meta, load_session, run_tool,
                        save_session)

logger = logging.getLogger("agent_core")
app = FastAPI()

# Allow wildcard origins locally to resolve file:// and port mismatch issues
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    user_message: str


@app.post("/new_session")
def api_new_session():
    sid, _ = create_new_session()
    return {"session_id": sid}


@app.get("/sessions")
def api_list_sessions():
    return {"sessions": list_sessions_with_meta()}


@app.get("/session/{session_id}")
def api_get_session(session_id: str):
    container = load_session(session_id)
    if container is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "metadata": container["metadata"],
        "messages": container["messages"],
    }


@app.delete("/session/{session_id}")
def api_del_session(session_id: str):
    p = get_session_path(session_id)
    if p.exists():
        p.unlink()
    return {"ok": True}


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    container = load_session(req.session_id)
    if container is None:
        raise HTTPException(status_code=404, detail="Session not found")

    history = container["messages"]
    history.append({"role": "user", "content": req.user_message})

    def sync_ollama_call():
        return ollama.chat(
            model="qwen2.5:7b-instruct",
            messages=history,
            options={"temperature": 0},
            stream=True,
        )

    async def async_iter_wrapper():
        # Execute blocking Ollama sync call in threadpool
        stream = await asyncio.to_thread(sync_ollama_call)
        full_text = ""

        for chunk in stream:
            piece = chunk["message"]["content"]
            full_text += piece
            yield json.dumps({"chunk": piece}) + "\n"
            await asyncio.sleep(0)  # Yield control back to event loop

        try:
            parsed = json.loads(full_text)
        except Exception:
            history.append({"role": "assistant", "content": full_text})
            save_session(req.session_id, history)
            yield json.dumps({"event": "done"}) + "\n"
            return

        if parsed.get("need_ask_user", False):
            history.append({"role": "assistant", "content": full_text})
            save_session(req.session_id, history)
            yield json.dumps(
                {"event": "ask_user", "question": parsed.get("question", "")}
            ) + "\n"
            return

        tool_name = parsed.get("tool_name")
        tool_args = parsed.get("tool_args", {})
        if tool_name:
            tool_res = run_tool(tool_name, tool_args)
            tool_record = f"[ToolCall] name={tool_name}, args={json.dumps(tool_args)}, result={json.dumps(tool_res)}"

            history.append({"role": "assistant", "content": full_text})
            history.append({"role": "tool", "content": tool_record})
            save_session(req.session_id, history)

            yield json.dumps({"event": "tool_result", "payload": tool_res}) + "\n"
            yield json.dumps({"event": "done"}) + "\n"
            return

        history.append({"role": "assistant", "content": full_text})
        save_session(req.session_id, history)
        yield json.dumps({"event": "done"}) + "\n"

    return StreamingResponse(async_iter_wrapper(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=5001)
