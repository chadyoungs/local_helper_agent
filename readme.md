# Local Helper Agent

A modern, local-first AI agent built on **Ollama + FastAPI**. It runs entirely on
your machine, uses a **harness sandbox** to isolate tools from the filesystem,
and follows a real **agentic loop** (`plan -> tool -> observe -> conclude`)
instead of one-shot tool dispatch.

## Highlights

- **Agentic loop** — the LLM can call tools multiple times in a single turn and
  see each result before concluding; a hard step budget prevents infinite loops.
- **Structured decisions, not free-form JSON** — the LLM returns a validated
  `AgentDecision` (ask / tool call / final answer) parsed by a Pydantic schema.
  Robust extraction strips markdown fences and prose; invalid output triggers
  bounded self-correction retries.
- **Native function calling (optional)** — `ENABLE_NATIVE_TOOLS=true` uses
  Ollama's tool-calling API; otherwise the deterministic structured-JSON path is
  used (more reliable for small local models).
- **Harness sandbox** — the LLM only ever emits *logical* filenames; absolute
  paths and `../` traversal are blocked, missing files return clean errors, and
  tool tracebacks never leak to the user.
- **Tool isolation** — each tool runs in a subprocess with a timeout.
- **Full front-end** — sidebar session manager (create / switch / rename /
  delete), drag-free file upload, live NDJSON streaming, Markdown rendering,
  collapsible tool-call panel, and an interactive "ask the user" modal.

## Project structure

```
local_helper_agent/
├── agent/                      # Core agent package
│   ├── settings.py             # Environment-driven configuration (.env supported)
│   ├── schemas.py              # Pydantic models: ToolSpec, AgentDecision, ToolResult
│   ├── tool_registry.py        # Typed tool registry + auto-generated LLM schema/prompt
│   ├── harness.py              # Sandbox: logical-name -> real-path mapping + traversal guard
│   ├── session_store.py        # Session persistence with per-session locking
│   ├── llm_client.py           # LLM abstraction (native tools + structured-JSON strategies)
│   ├── agent_loop.py           # The agentic loop (bounded, multi-step)
│   ├── tools_runner.py         # Subprocess tool runner with timeout
│   └── api.py                  # FastAPI app: sessions, files, NDJSON streaming chat
├── backend.py                  # Entry point (uvicorn)
├── index.html                  # Front-end (vanilla JS, no build step)
├── tools/                      # Standalone tool scripts (run in subprocesses)
│   ├── pdf_merge.py            # Merge PDFs to A4, optional blank pages
│   ├── pdf2img.py              # PDF -> JPEG per page (supports encrypted PDFs)
│   └── id_photo_bg_change.py   # Replace ID-photo background with a hex color
├── sessions/                   # Persisted chat sessions (JSON)
├── workspace/
│   ├── input/                  # Uploaded files (harness input root)
│   └── output/                 # Generated files (harness output root)
├── tests/                      # Unit + integration tests (no live Ollama needed)
├── statics/blank.pdf           # Built-in blank page asset for pdf_merge
├── .env.example                # Copy to .env and adjust
├── requirements.in / requirements.txt
└── requirements-dev.txt
```

## Requirements

- Python 3.10+
- Ollama installed and running locally (`ollama serve`)

## Quick start

```bash
# 1. Pull a model
ollama pull qwen2.5:7b-instruct

# 2. (Optional) configure
cp .env.example .env          # edit model / port / agent budget

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch
python3 backend.py            # -> http://127.0.0.1:5001

# 5. Open index.html in a browser
```

> **Note on `rembg`:** the ID-photo tool downloads an `.onnx` model from GitHub
> on first use (the `u2net` background-removal model).

## Configuration (`agent/settings.py` + `.env`)

| Variable | Default | Meaning |
| --- | --- | --- |
| `HOST` / `PORT` | `127.0.0.1` / `5001` | Bind address |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama service |
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct` | Model name |
| `TEMPERATURE` | `0` | Sampling temperature |
| `ENABLE_NATIVE_TOOLS` | `false` | Use Ollama native function calling |
| `MAX_AGENT_STEPS` | `5` | Max tool calls per turn |
| `JSON_RETRIES` | `2` | Structured-JSON self-correction retries |
| `TOOL_OUTPUT_MAX_CHARS` | `3000` | Tool result cap fed back to the model |
| `TOOL_TIMEOUT_SECONDS` | `120` | Per-tool subprocess timeout |
| `CORS_ORIGINS` | localhost set | Allowed browser origins |

## How the agent loop works

1. Your message is appended to session history.
2. The LLM returns a structured `AgentDecision`:
   - `need_ask_user=true` → an `ask_user` event; the turn pauses until you answer
     in the next message.
   - `tool_name` set → the tool runs inside the sandbox; its sanitized result is
     fed back as a `tool` message and the loop continues (up to `MAX_AGENT_STEPS`).
   - `final_answer` set → that is your reply; the turn ends.
3. If the LLM output cannot be parsed, it is retried with a self-correction hint;
   on exhaustion the agent asks you to rephrase.

## Harness sandbox behavior

- Tool arguments use **logical filenames only** — never absolute system paths.
- Bare input names (`2016.pdf`) resolve to `workspace/input/2016.pdf`.
- Output names resolve under `workspace/output/<first_input_stem>/`.
- Absolute paths and `../` traversal trigger a `HarnessError`; execution is blocked.
- Missing inputs return a clean error — no raw Python traceback reaches the UI.
- Each tool runs via `subprocess` (list argv, no shell), isolated from the agent.

## Adding a new tool

1. Write a standalone script under `tools/` (e.g. `tools/my_tool.py`).
2. Declare it as a `ToolSpec` in `agent/tool_registry.py`. Mark each parameter's
   `kind` as `"input"`, `"output"` or `"value"` so the harness maps paths for you.
3. Restart `backend.py`. The system prompt and LLM tool schema are generated
   automatically from the registry — no prompt editing required.

```python
ToolSpec(
    name="my_tool",
    description="What it does.",
    script_path=str(PROJECT_ROOT / "tools" / "my_tool.py"),
    parameters=[
        ParamSpec(flag="--input", type="str", required=True, kind="input",
                  desc="Input filename in workspace/input."),
        ParamSpec(flag="--output", type="str", required=True, kind="output",
                  desc="Output filename in workspace/output."),
    ],
)
```

## API endpoints

| Endpoint | Method | Description |
| --- | --- | --- |
| `/new_session` | POST | Create a chat session |
| `/sessions` | GET | List sessions (newest first) |
| `/session/{id}` | GET | Load a session (metadata + messages) |
| `/session/{id}` | PATCH | Rename a session |
| `/session/{id}` | DELETE | Delete a session |
| `/files` | GET | List files in `workspace/input` |
| `/upload` | POST | Upload a file into `workspace/input` (multipart) |
| `/chat` | POST | NDJSON streaming chat (agentic loop) |

### `/chat` NDJSON events

```
{"event":"user","content":"..."}
{"event":"tool_call","tool":"...","args":{...}}
{"event":"tool_result","tool":"...","payload":{...}}
{"event":"ask_user","question":"..."}
{"event":"chunk"|"final","content":"..."}
{"event":"error","message":"...","detail":"..."}
{"event":"done"}
```

## Tests

Unit and integration tests cover the harness sandbox, JSON extraction,
structured-decision parsing, the agent loop, and the HTTP API. They run without
a live Ollama (the LLM is mocked).

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```
