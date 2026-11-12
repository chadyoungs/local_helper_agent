# Local Helper Agent

A modern, local-first AI agent built on **FastAPI** and a **local LLM**. It runs
entirely on your machine, supports both **Ollama** and **LM Studio** (OpenAI
compatible) as inference backends, uses a **harness sandbox** to isolate tools
from the filesystem, and follows a real **agentic loop**
(`plan -> tool -> observe -> conclude`) instead of one-shot tool dispatch.

Runs on **Python 3.9 and newer**.

## Highlights

- **Pluggable inference backends** — run on **Ollama** or **LM Studio** with a
  single `.env` switch (`LLM_PROVIDER=ollama|lmstudio`); the same decision
  strategies work against either provider via a transport layer.
- **Agentic loop** — the LLM can call tools multiple times in a single turn and
  see each result before concluding; a hard step budget prevents infinite loops.
- **Structured decisions, not free-form JSON** — the LLM returns a validated
  `AgentDecision` (ask / tool call / final answer) parsed by a Pydantic schema.
  Robust extraction strips markdown fences and prose; invalid output triggers
  bounded self-correction retries.
- **Native function calling (optional)** — `ENABLE_NATIVE_TOOLS=true` uses the
  provider's tool-calling API; otherwise the deterministic structured-JSON path
  is used (more reliable for small local models).
- **Harness sandbox** — the LLM only ever emits *logical* filenames; absolute
  paths and `../` traversal are blocked, missing files return clean errors, and
  tool tracebacks never leak to the user.
- **Tool isolation** — each tool runs in a subprocess with a timeout.
- **Signature / fingerprint cutout** — removes the photographed paper background
  and returns a transparent RGBA PNG, with a locally-estimated paper level so
  uneven lighting and dark corners are not mistaken for ink.
- **Inline questions, no pop-ups** — when the agent needs clarification it
  appears as a message in the conversation stream and you answer in the message
  box; there is no modal window.
- **Every input file on screen** — the sidebar lists all of `workspace/input`,
  sub-folders included, with a refresh button; clicking a file inserts its name
  into the message box.

## Project structure

```
local_helper_agent/
├── agent/                      # Core agent package
│   ├── settings.py             # Environment-driven configuration (.env supported)
│   ├── schemas.py              # Pydantic models: ToolSpec, AgentDecision, ToolResult
│   ├── tool_registry.py        # Typed tool registry + auto-generated LLM schema/prompt
│   ├── harness.py              # Sandbox: logical-name -> real-path mapping + traversal guard
│   ├── session_store.py        # Session persistence with per-session locking
│   ├── llm_client.py           # LLM abstraction: transports (Ollama/LM Studio) + decision strategies
│   ├── agent_loop.py           # The agentic loop (bounded, multi-step)
│   ├── tools_runner.py         # Subprocess tool runner with timeout
│   └── api.py                  # FastAPI app: sessions, files, NDJSON streaming chat
├── backend.py                  # Entry point (uvicorn)
├── index.html                  # Front-end (vanilla JS, no build step)
├── tools/                      # Standalone tool scripts (run in subprocesses)
│   ├── pdf_merge.py            # Merge PDFs to A4, optional blank pages
│   ├── pdf2img.py              # PDF -> JPEG per page (supports encrypted PDFs)
│   ├── id_photo_bg_change.py   # Replace ID-photo background with a hex color
│   └── signature_fingerprint_cutout.py  # Signature + red fingerprint -> transparent PNG
├── sessions/                   # Persisted chat sessions (JSON)
├── workspace/
│   ├── input/                  # Input files (harness input root; sub-folders allowed)
│   └── output/                 # Generated files (harness output root)
├── tests/                      # Unit + integration tests (no live Ollama needed)
├── statics/blank.pdf           # Built-in blank page asset for pdf_merge
├── .env.example                # Copy to .env and adjust
├── requirements.txt            # Runtime deps (3.9+ compatible ranges)
├── requirements-extra.txt      # Optional heavy extras (rembg)
├── requirements.in             # Source list for pip-compile
└── requirements-dev.txt
```

## Requirements

- **Python 3.9+** (tested on 3.9 and 3.12+)
- One local inference backend:
  - **Ollama** installed and running (`ollama serve`), **or**
  - **LM Studio** with a model loaded and Developer mode enabled (exposes
    `http://127.0.0.1:1234/v1`; requires `pip install openai`)

## Quick start (Ollama)

```bash
# 1. Pull a model
ollama pull qwen2.5:7b-instruct

# 2. (Optional) configure
cp .env.example .env          # edit model / port / agent budget

# 3. Create an isolated environment and install dependencies.
#    Do NOT reuse a shared conda env: this project needs pydantic>=2.5 and
#    fastapi>=0.110, and upgrading a shared env can break other projects.
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# 4. Launch — always with the venv interpreter, because the tool subprocesses
#    inherit it (that is how they find numpy / opencv)
.venv/bin/python backend.py    # -> http://127.0.0.1:5001

# 5. Open index.html in a browser
```

## Quick start (LM Studio)

1. In LM Studio, load a model, then click **Developer** (bottom-right) so the
   OpenAI-compatible server turns on at `http://127.0.0.1:1234/v1`.
2. Install the OpenAI client and configure:

```bash
pip install openai
# .env:
#   LLM_PROVIDER=lmstudio
#   LMSTUDIO_BASE_URL=http://127.0.0.1:1234
#   OLLAMA_MODEL=<exact model name from LM Studio's Developer tab>
python3 backend.py
```

> **Note on `rembg`:** the ID-photo tool needs `rembg`, which downloads an
> `.onnx` model from GitHub on first use. It is kept out of the base install:
> `python3 -m pip install -r requirements-extra.txt`.

## Inference backends

The LLM layer is split into **transports** (provider-specific API adapters) and
**strategies** (provider-agnostic decision logic):

- `OllamaTransport` — native `/api/chat` (Ollama package).
- `LMStudioTransport` — OpenAI-compatible `/v1/chat/completions` (OpenAI package,
  imported lazily; JSON mode uses `response_format={"type":"json_object"}`).
- `JSONStructuredLLMClient` / `NativeToolLLMClient` — run unchanged on either
  backend. `build_transport(settings)` picks the transport from `LLM_PROVIDER`.

Switch with a single `.env` line; the agent loop and tools are identical.

## Configuration (`agent/settings.py` + `.env`)

| Variable | Default | Meaning |
| --- | --- | --- |
| `HOST` / `PORT` | `127.0.0.1` / `5001` | Bind address |
| `LLM_PROVIDER` | `ollama` | `ollama` or `lmstudio` |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama service |
| `LMSTUDIO_BASE_URL` | `http://127.0.0.1:1234` | LM Studio / OpenAI-compatible server |
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct` | Model name (both providers) |
| `TEMPERATURE` | `0` | Sampling temperature |
| `ENABLE_NATIVE_TOOLS` | `false` | Use native function calling when supported |
| `MAX_AGENT_STEPS` | `5` | Max tool calls per turn |
| `JSON_RETRIES` | `2` | Structured-JSON self-correction retries |
| `TOOL_OUTPUT_MAX_CHARS` | `3000` | Tool result cap fed back to the model |
| `TOOL_TIMEOUT_SECONDS` | `120` | Per-tool subprocess timeout |
| `TOOL_PYTHON` | the running interpreter | Interpreter used to launch tool scripts; defaults to `sys.executable` so tools see the same numpy/opencv as the server |
| `CORS_ORIGINS` | localhost set | Allowed browser origins |

## File access model

Inputs come from one place only: copy your files into `workspace/input` (plain
files or sub-folders), then reference them **by name in chat**.

- The sidebar lists **every** file under `workspace/input`, recursively,
  grouped by folder, with a size and a reload button (⟳). Hidden helper files
  such as `.gitkeep` are skipped. Clicking a row inserts its name into the
  message box.
- Files in sub-folders are addressed by their **relative path** exactly as
  listed, e.g. `2024/scan.jpg`.
- **Fuzzy matching**: you can use a shortened or partial filename (e.g. `2016`
  for `2016届毕业典礼.pdf`); the harness resolves it to the unique matching
  file — sub-folders included — and lists candidates when ambiguous.
- Absolute system paths and `../` traversal are always blocked by the sandbox.

Outputs are written under `workspace/output/<chat-title>_<timestamp>/` (each
turn gets its own timestamped folder). Tools never modify input files —
`pdf_merge` copies its inputs to a staging dir before merging.

## Signature / fingerprint cutout

`tools/signature_fingerprint_cutout.py` turns a phone photo of a signed sheet
into a transparent PNG that keeps only the black signature and the red
fingerprint.

How it works:

1. **Local paper estimate** — a large median filter (computed on a downscaled
   copy) estimates the paper brightness *per region*, so a vignette or a shadow
   is not carved out as "ink".
2. **Black ink** — relative darkness `1 - brightness / paper` mapped through a
   smooth ramp, which keeps anti-aliased stroke edges.
3. **Red ink** — HSV **saturation** plus a red hue gate. Saturation separates
   the stamp from grey/yellow paper far better than "red minus green".
4. **Solid print** — the paper gaps inside a stamp/fingerprint are closed and
   repainted with the ink's own median colour (painting them with the paper
   colour would leave a pale blob).
5. **Clean-up** — speck removal by connected-component area, optional Gaussian
   feathering, alpha cutoff, then cropping to the ink bounding box using a
   strong threshold (so stray noise cannot defeat the crop).
6. **Write-out** — saved with Pillow as RGBA PNG (OpenCV's `imwrite` would read
   RGBA as BGRA and silently swap red and blue).

The script prints exactly one JSON object on stdout, for success **and** for
failure, e.g.:

```json
{"ok": true, "output": ".../signature_cutout.png",
 "image": {"width": 823, "height": 344},
 "ink_pixels": 42680, "ink_ratio": 0.15075,
 "has_black_ink": true, "has_red_ink": true}
```

Tuning flags (`--ink-min`, `--ink-full`, `--red-min`, `--red-full`, `--cutoff`)
are optional and normally left alone; the defaults handle uneven lighting. A
`.jpg` output name is silently corrected to `.png` (JPEG has no alpha channel)
and the JSON `note` says so.

## How the agent loop works

1. Your message is appended to session history.
2. The LLM returns a structured `AgentDecision`:
   - `need_ask_user=true` → an `ask_user` event; the turn pauses until you
     answer in the next message.
   - `tool_name` set → the tool runs inside the sandbox; its sanitized result is
     fed back as a `tool` message and the loop continues (up to `MAX_AGENT_STEPS`).
   - `final_answer` set → that is your reply; the turn ends.
3. If the LLM output cannot be parsed, it is retried with a self-correction hint;
   on exhaustion the agent asks you to rephrase.

Two pieces of live context are injected on every decision, because a local model
otherwise guesses:

- **The real `workspace/input` listing** (relative names + size), appended to the
  system prompt. The tool catalog is embedded in the system prompt too, so the
  model never has to ask whether a tool exists.
- **A compact, verdict-first tool result** — `STATUS: OK` / `STATUS: FAILED
  (exit N)` first, then a one-paragraph digest with `workspace/...` relative
  paths, and a closing "do not repeat this call" instruction.

Loop guards (they keep a small model from burning the whole budget on itself):

| Situation | Behaviour |
| --- | --- |
| Same tool + same args already **succeeded** | Not executed again; the turn ends with the produced file |
| Same tool + same args **failed** and is repeated | Turn stops and asks you for the exact value instead of retrying blindly |
| Same tool attempted 4+ times in one turn | Ends with the latest successful result |
| Budget exhausted after a success | Reports the produced file, not a bare "budget exceeded" |

## Front-end behaviour

- Sidebar: session manager (create / switch / rename / delete) plus the input
  file tree described above.
- Clarifications from the agent are rendered **inline in the conversation**
  with a "Needs your input" badge and an amber left border; the message box
  placeholder shows the question. There is no modal/pop-up window — you just
  keep typing.
- Live NDJSON streaming, Markdown rendering, collapsible tool-call panel, and
  filename autocomplete in the chat box (works for sub-folder files too).

## Harness sandbox behavior

- Tool arguments use **logical filenames only** — never absolute system paths.
- Bare input names (`2016.pdf`) resolve to `workspace/input/2016.pdf`.
- Nested names (`2024/scan.jpg`) resolve to `workspace/input/2024/scan.jpg`.
- Output names resolve under `workspace/output/<first_input_stem>/`.
- Absolute paths and `../` traversal trigger a `HarnessError`; execution is blocked.
- Missing inputs return a clean error — no raw Python traceback reaches the UI.
- Each tool runs via `subprocess` (list argv, no shell), isolated from the agent.

## Python 3.9 compatibility

The code base targets **3.9+**, so two rules apply:

- **Annotations must not use PEP 604 unions** (`str | None`). They parse on 3.9
  but blow up at runtime when something evaluates them (Pydantic, FastAPI,
  `dataclasses`). Use `Optional[str]` / `Union[...]`.
- **No 3.10+ syntax** (`match`, multi-item parenthesised `with`, ...) and no
  3.11+ APIs (`tomllib`, `typing.Self`, `ExceptionGroup`).

`tests/test_py39_syntax.py` enforces both without needing a 3.9 interpreter:
every module is parsed with `feature_version=(3, 9)` and every annotation is
scanned for `|`. Dependency ranges in `requirements.txt` are deliberately
range-based so the same file installs on 3.9 and on newer interpreters.

## Troubleshooting

**The UI says "failed to create session".** That message means the browser could
not reach the backend — start it and watch the terminal:

```bash
.venv/bin/python backend.py
```

If it exits immediately, the message on stderr says why. Two common causes:

- **`pydantic 1.x` / `fastapi 0.85` in the current environment** — the project
  uses the Pydantic v2 API, so `backend.py` now prints an explicit hint instead
  of failing with `has no attribute 'model_validate'`. Create the venv shown in
  Quick start rather than upgrading a shared conda env.
- **An outdated `openai` package with `LLM_PROVIDER=lmstudio`** — the app still
  starts (sessions and the file panel work); `/chat` returns an error event
  telling you to `pip install -U "openai>=1.0"`. The legacy 0.x SDK is also
  supported for plain JSON chat.

**A tool reports `ModuleNotFoundError: No module named 'cv2'`.** The tool ran on
a different interpreter than the server. Tools default to `sys.executable`; if
you override `TOOL_PYTHON`, make sure it has `numpy` + `opencv-python-headless`
installed.

**The session list is empty or the app logs "ignoring unreadable session
file".** A session JSON was truncated (usually by a crash or a full disk). Those
files are skipped instead of breaking the list; delete them from `sessions/` to
silence the warning.

**The agent runs the same tool until `I stopped after N steps...`.** It was
guessing: without the real input listing it invents filenames, each miss costs a
step, and the budget runs out. Check that the system prompt carries both blocks:

```bash
.venv/bin/python -c "from agent.tool_registry import SYSTEM_PROMPT; \
print(len(SYSTEM_PROMPT), '{tool_list_text}' in SYSTEM_PROMPT)"
```

The placeholder must be **False** and the length should be a few thousand
characters. A repeated identical call now short-circuits (see "How the agent
loop works"), so a genuine 5-step loop means the model is still missing context.

**The agent asks "Do you have access to a tool that can ...?"** That question is
a symptom, not a feature: the tool catalog never reached the prompt. `SYSTEM_PROMPT`
is an f-string, so `{{tool_list_text}}` rendered as a literal `{tool_list_text}`
and the model only knew tool names from the "Special cases" prose. It is now
substituted once at import (`test_system_prompt_carries_the_full_tool_catalog`
guards against a regression). If you edit the prompt template, never hand a
placeholder to the model — substitute it, or the model will ask you about
capabilities you cannot change.

## Adding a new tool

1. Write a standalone script under `tools/` (e.g. `tools/my_tool.py`) that
   prints exactly one JSON object on stdout (success and failure) and never
   emits a traceback.
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
| `/files` | GET | List **every** file in `workspace/input` recursively |
| `/chat` | POST | NDJSON streaming chat (agentic loop) |

`GET /files` returns one entry per file, sorted by relative path:

```json
{"count": 2,
 "files": [{"name": "2024/scan.jpg", "dir": "2024", "size": 91342},
           {"name": "top.pdf", "dir": "", "size": 20481}]}
```

`name` is the relative path, i.e. exactly the logical filename a tool argument
accepts.

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

Unit and integration tests cover the harness sandbox (including nested input
paths), JSON extraction, structured-decision parsing, both inference transports
(Ollama + LM Studio), the agent loop, the HTTP API, the recursive file listing,
the signature cutout pipeline (run end-to-end on a synthesised photo and
verified pixel by pixel), the agent loop guards (duplicate-call short-circuit,
repeat-failure escalation, `workspace/...` path relativisation), the full tool
catalog being present in the system prompt, the 3.9 syntax/annotation guards, and
the front-end requirements (no modal, full input listing). They run without a
live model.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

The cutout test needs `numpy`, `opencv-python-headless` and `Pillow`; it is
skipped automatically when they are missing.
