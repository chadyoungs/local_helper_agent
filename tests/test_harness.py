"""Tests for the harness sandbox (logical-name mapping + traversal blocking)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.harness import HarnessError, resolve_input, rewrite_tool_arguments, split_logical_names
from agent.settings import Settings
from agent.tool_registry import TOOL_REGISTRY

# A throwaway workspace so tests never touch the real one.
TEST_ROOT = Path(__file__).resolve().parent / "_test_ws"
INPUT = TEST_ROOT / "workspace" / "input"
OUTPUT = TEST_ROOT / "workspace" / "output"


def _settings() -> Settings:
    s = Settings(workspace_root=TEST_ROOT / "workspace")
    s.ensure_dirs()
    return s


def _tool(name: str):
    return next(t for t in TOOL_REGISTRY if t.name == name)


@pytest.fixture(autouse=True)
def _ws():
    INPUT.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (INPUT / "a.pdf").write_bytes(b"%PDF-1.4 fake")
    (INPUT / "b.pdf").write_bytes(b"%PDF-1.4 fake")
    yield
    # leave fixture files in place for further tests


# --------------------------------------------------------------------------- #
def test_split_logical_names():
    assert split_logical_names("a.pdf b.pdf") == ["a.pdf", "b.pdf"]
    assert split_logical_names("a.pdf,b.pdf") == ["a.pdf", "b.pdf"]
    assert split_logical_names("  x.pdf , y.pdf  ") == ["x.pdf", "y.pdf"]


def test_resolve_input_bare_filename():
    settings = _settings()
    resolved = resolve_input("a.pdf", settings)
    assert resolved == (INPUT / "a.pdf").resolve()
    assert resolved.exists()


def test_resolve_input_rejects_absolute():
    settings = _settings()
    with pytest.raises(HarnessError):
        resolve_input("/etc/passwd", settings)


def test_resolve_input_rejects_traversal():
    settings = _settings()
    with pytest.raises(HarnessError):
        resolve_input("../../etc/passwd", settings)


def test_resolve_input_missing_file():
    settings = _settings()
    with pytest.raises(HarnessError):
        resolve_input("does_not_exist.pdf", settings)


def test_pdf_merge_inputs_stay_single_string():
    settings = _settings()
    tool = _tool("pdf_merge")
    rewritten = rewrite_tool_arguments(
        tool,
        {"inputs": "a.pdf b.pdf", "blank-count": 1, "output": "merged.pdf"},
        settings,
    )
    # still a single space-joined string, now with absolute paths
    assert isinstance(rewritten["inputs"], str)
    parts = rewritten["inputs"].split(" ")
    assert len(parts) == 2
    assert all(Path(p).exists() for p in parts)


def test_pdf_merge_list_inputs_normalized():
    settings = _settings()
    tool = _tool("pdf_merge")
    rewritten = rewrite_tool_arguments(
        tool, {"inputs": ["a.pdf", "b.pdf"], "output": "merged.pdf"}, settings
    )
    assert isinstance(rewritten["inputs"], str)
    assert len(rewritten["inputs"].split(" ")) == 2


def test_output_mapped_under_primary_stem():
    settings = _settings()
    tool = _tool("pdf_merge")
    rewritten = rewrite_tool_arguments(
        tool, {"inputs": "a.pdf", "output": "merged.pdf"}, settings
    )
    # output lives under output/<input_stem>/merged.pdf
    assert rewritten["output"].endswith("a/merged.pdf")
    assert Path(rewritten["output"]).parent.exists()


def test_value_params_passed_through():
    settings = _settings()
    tool = _tool("id_photo_bg_change")
    rewritten = rewrite_tool_arguments(
        tool,
        {"input": "a.pdf", "output": "out.png", "target_color": "#FF0000"},
        settings,
    )
    assert rewritten["target_color"] == "#FF0000"
