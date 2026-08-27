from pathlib import Path

# ========== 路径配置 ==========
PROJECT_ROOT = Path(__file__).parent.resolve()
WORKSPACE = PROJECT_ROOT / "workspace"
WORKSPACE_INPUT = WORKSPACE / "input"
WORKSPACE_OUTPUT = WORKSPACE / "output"
SESSIONS_DIR = PROJECT_ROOT / "sessions"

# 自动建目录
WORKSPACE_INPUT.mkdir(parents=True, exist_ok=True)
WORKSPACE_OUTPUT.mkdir(parents=True, exist_ok=True)
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# ========== Tool 注册表，在这里增删改工具 ==========
TOOL_REGISTRY = [
    {
        "name": "pdf_merge",
        "description": "Merge multiple PDF files into one single PDF file.",
        "script_path": str(PROJECT_ROOT / "tools" / "pdf_merge.py"),
        "parameters": [
            {
                "flag": "--inputs",
                "type": "list[str]",
                "required": True,
                "desc": "List of input PDF filenames inside workspace/input, support both comma-seperated and space-seperated ",
            },
            {
                "flag": "--blank-count",
                "type": "int",
                "required": False,
                "desc": "Number of blank pages inserted between pdf documents, 0 for no blank pages",
            },
            {
                "flag": "--output",
                "type": "str",
                "required": True,
                "desc": "Output PDF filename saved to workspace/output",
            },
        ],
    },
    {
        "name": "id_photo_bg_change",
        "description": "Change background color of an id image file.",
        "script_path": str(PROJECT_ROOT / "tools" / "id_photo_bg_change.py"),
        "parameters": [
            {
                "flag": "--input",
                "type": "str",
                "required": True,
                "desc": "Input image filename inside workspace/input",
            },
            {
                "flag": "--output",
                "type": "str",
                "required": True,
                "desc": "Output image filename saved to workspace/output",
            },
            {
                "flag": "--target-color",
                "type": "str",
                "required": True,
                "desc": "Hex color code e.g. Red:#FF0000, Blue:#0000FF, White:#FFFFFF",
            },
        ],
    },
    {
        "name": "pdf2img",
        "description": "Convert multiple PDF files to JPEG images, each page becomes one jpeg. Some PDFs may be password-protected. If password missing, ask user for password of specific pdf file.",
        "script_path": str(PROJECT_ROOT / "tools" / "pdf2img.py"),
        "parameters": [
            {
                "flag": "--inputs",
                "type": "list[str]",
                "required": True,
                "desc": "List of input PDF filenames inside workspace/input",
            },
            {
                "flag": "--output",
                "type": "str",
                "required": True,
                "desc": "Output folder name under workspace/output to store jpegs",
            },
            {
                "flag": "--passwords",
                "type": "list[str]",
                "required": False,
                "desc": "Password list for each pdf; use __NO_PWD__ placeholder for pdf without password. Must match order of inputs list.",
            },
        ],
    },
]

SYSTEM_PROMPT_TPL = """
You are a tool dispatcher. You must ONLY output valid JSON, NO extra explanation text outside JSON block.

# MANDATORY RULE FOR pdf_merge (VIOLATION CAUSES CLI FAILURE)
For tool pdf_merge:
- "inputs" MUST be ONE single string. Multiple filenames are separated by SPACE or COMMA inside this string.
- NEVER output "inputs" as JSON array []. Array will cause tool runtime error.
✅ CORRECT: {"inputs":"fileA.pdf fileB.pdf","blank-count":1,"output":"out.pdf"}
❌ FORBIDDEN: {"inputs":["fileA.pdf","fileB.pdf"]}
- blank‑count: optional integer, number of blank pages inserted between documents.
- blank‑pdf argument DOES NOT exist; blank page resource is built‑in.

General Rules:
1. All file‑related tool parameters use logical filenames only, DO NOT fill absolute system paths like /home or C:/.
2. When user gives a bare filename (no slashes), use that bare filename in tool arguments.
   The underlying harness layer will automatically map input files to `workspace/input/<filename>`, and outputs to `workspace/output/`.
3. Never fabricate file paths. If you are unsure of filename, ask user for clarification.
4. Do not manually concatenate workspace/input prefix in toolcall arguments. Keep arguments as logical names.

Special rule for id_photo_bg_change:
Important: For the `input` argument, ALWAYS use the real uploaded image filename from chat context.
DO NOT hard‑code "photo.jpg".
Examples:
User:"change my uploaded img_001.jpg background to red"
tool_args: {"input":"img_001.jpg","output":"id_photo_result","target_color":"#FF0000"}
User:"把我上传的证件照20260827.jpg换成蓝底"
tool_args: {"input":"20260827.jpg","output":"id_photo_result","target_color":"#0000FF"}

Special rule for pdf2img:
When converting password‑encrypted PDF files, if you do NOT have the password for a specific pdf file, set need_ask_user=true and explicitly ask user to provide password for that exact pdf filename.
Do NOT guess passwords. Use "__NO_PWD__" only for documents confirmed password‑free.

Available tools:
{tool_list_text}

Response JSON schema:
{{
  "need_ask_user": boolean,
  "question": "string or empty",
  "tool_name": "tool name from registry or empty string",
  "tool_args": {{}}
}}

Logic:
1. If user request lacks required information (including missing pdf password): set need_ask_user=true, fill question, others empty.
2. If you can call tool: need_ask_user=false, fill tool_name and tool_args.
3. tool_args key name is parameter flag without "--".
Most multi‑item parameters: use JSON array.
EXCEPTION ONLY for pdf_merge inputs: use single space/comma separated string, NOT array.
For optional passwords argument: use "__NO_PWD__" as placeholder for pdf with no password.
Do NOT output markdown ```json fence. Output pure JSON only.
"""


