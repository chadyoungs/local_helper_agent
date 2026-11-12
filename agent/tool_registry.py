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
    ToolSpec(
        name="signature_fingerprint_cutout",
        description=(
            "Cut a signature and a red fingerprint out of a photographed sheet: remove the "
            "paper background (grey, yellowish, uneven lighting, vignetted corners) and keep "
            "only the black ink strokes and the red fingerprint, producing a "
            "transparent-background PNG (RGBA). The ink keeps its original colour."
        ),
        script_path=str(PROJECT_ROOT / "tools" / "signature_fingerprint_cutout.py"),
        parameters=[
            ParamSpec(
                flag="--input",
                type="str",
                required=True,
                kind="input",
                desc="Input photo filename inside workspace/input (the real uploaded filename).",
            ),
            ParamSpec(
                flag="--output",
                type="str",
                required=True,
                kind="output",
                desc=(
                    "Output PNG filename under workspace/output. Always keep the .png "
                    "extension: a transparent result cannot be stored as JPEG."
                ),
            ),
            ParamSpec(
                flag="--ink-min",
                type="int",
                required=False,
                kind="value",
                desc=(
                    "Darkness percent where black ink starts to appear (default 8). "
                    "Lower it for faint pencil, raise it to ignore paper shadows."
                ),
            ),
            ParamSpec(
                flag="--ink-full",
                type="int",
                required=False,
                kind="value",
                desc="Darkness percent where black ink becomes fully opaque (default 55).",
            ),
            ParamSpec(
                flag="--red-min",
                type="int",
                required=False,
                kind="value",
                desc="Saturation percent where red ink starts to appear (default 18). Lower it for a faint stamp.",
            ),
            ParamSpec(
                flag="--red-full",
                type="int",
                required=False,
                kind="value",
                desc="Saturation percent where red ink becomes fully opaque (default 45).",
            ),
            ParamSpec(
                flag="--cutoff",
                type="int",
                required=False,
                kind="value",
                desc="Drop alpha below this percent to clean faint paper residue (default 4).",
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


_SYSTEM_PROMPT_TEMPLATE = f"""
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
   password or two different files that both match), set need_ask_user=true,
   fill question, and leave everything else empty.
2. Else if a tool call resolves the request, set need_ask_user=false, fill tool_name
   and tool_args, and leave final_answer empty.
3. Else (the request is answered without a tool, or you are concluding after tool
   results were already observed), set final_answer to your final, natural-language
   answer and leave tool fields empty.
Asking the user should be the last resort: the file panel already shows every
available input file, so pick the right one from there instead of asking.
The tool catalog above is COMPLETE and FIXED. Never ask the user whether a tool
exists, whether they "have access" to one, or how to install dependencies --
that is not something they can answer. If a request matches a listed tool, call
it. If no listed tool fits, say plainly that this build cannot do it.

# File handling rules
1. All file-related tool arguments use LOGICAL filenames only. Never output
   absolute system paths such as /home or C:\\
2. The harness maps bare input filenames to workspace/input/<filename> and output
   names to workspace/output/. Files that live in a sub-folder are referenced by
   their relative path exactly as shown in the file panel (e.g. "2024/scan.jpg").
   Never concatenate workspace/input manually.
3. Use the exact filename as it appears in the file panel; never fabricate or
   guess a filename. If unsure, ask the user.
4. tool_args keys are the parameter flags without leading dashes (e.g. "inputs").
5. Outputs are always written under workspace/output; tools never modify input
   files (pdf_merge copies inputs before merging).

# Tool-call discipline
6. Once a tool succeeds and produces the requested output, respond with the
   final answer referencing that output. Do NOT call the same tool again with
   the same arguments.
7. Optional tuning parameters exist but should be OMITTED by default; the tools
   have sensible values. Only pass them when the user asks for a different result
   or when a first attempt produced obviously wrong output.
8. REQUIRED parameters must ALWAYS be sent on the first call -- never leave one
   out to "fill it in later". Every file-producing tool needs an output filename
   (e.g. "merged.pdf"); without it the call fails and wastes a step.

# Special cases
- pdf_merge "inputs" MUST be ONE string with space- or comma-separated filenames.
  Never a JSON array. Example: {{"inputs":"fileA.pdf fileB.pdf","blank-count":1,"output":"out.pdf"}}
- id_photo_bg_change "input" MUST be the real uploaded filename from chat context.
  Never hard-code "photo.jpg".
- pdf2img "passwords" must match input order; use "__NO_PWD__" for unencrypted PDFs.
- signature_fingerprint_cutout handles the whole "把签名和指纹抠出来 / cut out my
  signature and fingerprint / remove the paper background" family of requests:
  call it directly with the photo filename and a .png output name
  (e.g. "signature_cutout.png"). Never ask which part is the signature, never ask
  for the paper colour, and never suggest .jpg/.jpeg for this tool -- it writes an
  RGBA image whose background is fully transparent. It accepts photos with uneven
  lighting and dark corners, and works even when only a black signature or only a
  red fingerprint is present.

# Available tools
{{tool_list_text}}
""".strip()

# NOTE: an f-string turns ``{{tool_list_text}}`` into a literal ``{tool_list_text}``
# and nothing substituted it, so for a long time the model received the prompt
# WITHOUT any tool catalog: it picked the wrong tool, omitted required flags, and
# asked the user "do you have access to a tool that can ...?". Build the real
# prompt here, once, after ``tool_list_text()`` is defined.
SYSTEM_PROMPT = _SYSTEM_PROMPT_TEMPLATE.replace("{tool_list_text}", tool_list_text())
