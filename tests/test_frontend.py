"""Static guards for the two frontend requirements.

* clarifications are rendered inside the conversation -- no pop-up/modal window
* the file panel lists every file under workspace/input, with a refresh control
"""

from __future__ import annotations

from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
INDEX = PROJECT / "index.html"


def _html() -> str:
    return INDEX.read_text(encoding="utf-8")


def test_no_ask_user_modal_window():
    html = _html()
    # the modal markup, its CSS and its JS helpers must all be gone
    for token in ("askMask", "askInput", "modal-mask", "showAsk", "Clarification needed"):
        assert token not in html, "leftover ask-user modal piece: %s" % token


def test_clarification_rendered_inline():
    html = _html()
    assert "function addAsk(" in html
    assert 'ev.event === "ask_user"' in html
    assert "msg ask" in html  # inline bubble style


def test_file_panel_lists_all_input_files():
    html = _html()
    assert "/files" in html
    assert "fileCount" in html
    assert "btnRefreshFiles" in html
    # grouped rendering for sub-folders
    assert "file-group" in html
    assert "f.dir" in html
