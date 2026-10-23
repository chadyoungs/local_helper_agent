"""Typed tool registry and prompt/schema generation.

Tools are declared once as ``ToolSpec`` objects. Everything the LLM sees --
the system-prompt tool list and the native function-calling schema -- is
generated from this registry, so the two can never drift apart.
"""

from __future__ import annotations

from pathlib import Path

from .schemas import ParamSpec, ToolSpec
from .settings import PROJECT_ROOT

# --------------------------------------------------------------------------- #
# Registry: add / remove tools here
# --------------------------------------------------------------------------- #
TOOL_REGISTRY: list[ToolSpec] = [
    ToolSpec(
        name="pdf_merge",
        description=(
            "Merge multiple PDF files into one single PDF file, forcing every "
            "page to A4 portrait. Optional blank pages can be inserted between documents."
        ),
        script_path=str(PROJECT_ROOT / "tools" / "pdf_merge.py"),
        parameters=[
            ParamSpec(
                flag="--inputs",
                type="list[str]",
                required=True,
                kind="input",
                multi=True,
                desc=(
                    "Input PDF filenames inside workspace/input. For this tool the "
                    "value is ONE string with filenames separated by space or comma "
                    "(never a JSON array)."
                ),
            ),
            ParamSpec(
                flag="--blank-count",
                type="int",
                required=False,
                kind="value",
                desc="Number of blank pages inserted between documents, 0 for none.",
            ),
            ParamSpec(
                flag="--output",
                type="str",
                required=True,
                kind="output",
                desc="Output PDF filename saved to workspace/output.",
            ),
        ],
    ),
    ToolSpec(
        name="id_photo_bg_change",
        description="Change the background color of an ID (portrait) image to a solid hex color.",
        script_path=str(PROJECT_ROOT / "tools" / "id_photo_bg_change.py"),
        parameters=[
            ParamSpec(
                flag="--input",
                type="str",
                required=True,
                kind="input",
                desc="Input image filename inside workspace/input (use the real uploaded filename).",
            ),
            ParamSpec(
                flag="--output",
                type="str",
                required=True,
                kind="output",
                desc="Output image filename saved to workspace/output.",
            ),
            ParamSpec(
                flag="--target-color",
                type="str",
                required=True,
                kind="value",
                desc="Hex color code, e.g. #FF0000 (red), #0000FF (blue), #FFFFFF (white).",
            ),
        ],
    ),
    ToolSpec(
        name="pdf2img",
        description=(
            "Convert PDF files to JPEG images, one JPEG per page. Some PDFs may be "
            "password-protected; if the password is missing, ask the user."
        ),
        script_path=str(PROJECT_ROOT / "tools" / "pdf2img.py"),
        parameters=[
            ParamSpec(
                flag="--inputs",
                type="list[str]",
                required=True,
                kind="input",
                multi=True,
                desc="Input PDF filenames inside workspace/input.",
            ),
            ParamSpec(
                flag="--output",
                type="str",
                required=True,
                kind="output",
                desc="Output folder name under workspace/output to store JPEGs.",
            ),
            ParamSpec(
                flag="--passwords",
                type="list[str]",
                required=False,
                kind="value",
                multi=True,
                desc=(
                    "Password per input PDF, must match the order of inputs. Use "
                    "__NO_PWD__ placeholder for a PDF without password."
                ),
            ),
        ],
    ),
]


# --------------------------------------------------------------------------- #
# Derived artifacts
# --------------------------------------------------------------------------- #
def tool_schemas_for_llm() -> list[dict]:
    """Function-calling schema objects for Ollama's native tools parameter."""
    return [t.llm_schema() for t in TOOL_REGISTRY]


def tool_list_text() -> str:
    """Human-readable tool descriptions embedded into the system prompt."""
    blocks = []
    for t in TOOL_REGISTRY:
        blocks.append(
            f"Tool name: {t.name}\n"
            f"Description: {t.description}\n"
            f"Parameters:\n{t.cli_schema_text()}"
        )
    return "\n---\n".join(blocks)


SYSTEM_PROMPT = f"""
You are a local tool-dispatcher agent. You respond ONLY with a single JSON object
conforming to the schema below. Do NOT output markdown fences, commentary, or
anything outside the JSON object.

# Response JSON schema
{{
  "need_ask_user": boolean,
  "question": "string or empty",
  "tool_name": "name of an available tool or empty string",
  "tool_args": {{ "arg_key_without_dashes": value }},
  "final_answer": "string or empty"
}}

# Decision rules (evaluated in order)
1. If the request lacks information you cannot guess (including a missing PDF
   password), set need_ask_user=true, fill question, and leave everything else empty.
2. Else if a tool call resolves the request, set need_ask_user=false, fill tool_name
   and tool_args, and leave final_answer empty.
3. Else (the request is answered without a tool, or you are concluding after tool
   results were already observed), set final_answer to your final, natural-language
   answer and leave tool fields empty.

# File handling rules
1. All file-related tool arguments use LOGICAL filenames only. Never output
   absolute system paths such as /home or C:\\.
2. The harness maps bare input filenames to workspace/input/<filename> and output
   names to workspace/output/. Never concatenate workspace/input manually.
3. Never fabricate filenames. If unsure, ask the user.
4. tool_args keys are the parameter flags without leading dashes (e.g. "inputs").

# Special cases
- pdf_merge "inputs" MUST be ONE string with space- or comma-separated filenames.
  Never a JSON array. Example: {{"inputs":"fileA.pdf fileB.pdf","blank-count":1,"output":"out.pdf"}}
- id_photo_bg_change "input" MUST be the real uploaded filename from chat context.
  Never hard-code "photo.jpg".
- pdf2img "passwords" must match input order; use "__NO_PWD__" for unencrypted PDFs.

# Available tools
{{tool_list_text}}
""".strip()
