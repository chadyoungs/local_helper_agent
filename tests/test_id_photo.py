"""Tests for tools/id_photo_bg_change.py.

The model-free fallback is what runs on a plain install (no rembg), so that is
what these tests exercise: a solid studio background must be replaced exactly,
while the subject keeps its own colours.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2", reason="opencv is only needed by the image tools")
np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

PROJECT = Path(__file__).resolve().parent.parent
TOOL = PROJECT / "tools" / "id_photo_bg_change.py"


def _portrait(path: Path) -> None:
    img = np.zeros((600, 420, 3), np.uint8)
    img[:, :] = (60, 90, 160)  # solid blue backdrop
    cv2.ellipse(img, (210, 330), (95, 150), 0, 0, 360, (215, 180, 165), -1, cv2.LINE_AA)
    cv2.ellipse(img, (210, 200), (80, 70), 0, 0, 360, (40, 32, 28), -1, cv2.LINE_AA)
    Image.fromarray(img).save(str(path), quality=92)


@pytest.fixture()
def portrait(tmp_path) -> Path:
    p = tmp_path / "portrait.jpg"
    _portrait(p)
    return p


def test_background_replaced_and_subject_kept(portrait, tmp_path):
    out = tmp_path / "red.png"
    proc = subprocess.run(
        [sys.executable, str(TOOL), "--input", str(portrait),
         "--output", str(out), "--target-color", "#FF0000"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(proc.stdout.strip())
    assert report["ok"] is True
    assert report["method"] in {"rembg", "solid-background-fallback"}

    rgb = np.asarray(Image.open(report["output"]).convert("RGB")).astype(int)
    h, w = rgb.shape[:2]
    corner = rgb[3, 3]
    assert corner[0] > 245 and corner[1] < 10 and corner[2] < 10, corner
    # the face keeps its skin tone instead of being painted over
    face = rgb[h // 2, w // 2]
    assert face[1] > 120 and face[2] > 100, face


def test_invalid_colour_reports_clean_json_error(portrait, tmp_path):
    proc = subprocess.run(
        [sys.executable, str(TOOL), "--input", str(portrait),
         "--output", str(tmp_path / "x.png"), "--target-color", "red"],
        capture_output=True, text=True,
    )
    report = json.loads(proc.stdout.strip())
    assert proc.returncode != 0 and report["ok"] is False
    assert "Traceback" not in proc.stderr


def test_output_file_target_is_not_turned_into_a_folder(portrait, tmp_path):
    out = tmp_path / "out.jpg"
    proc = subprocess.run(
        [sys.executable, str(TOOL), "--input", str(portrait),
         "--output", str(out), "--target-color", "#0000FF"],
        capture_output=True, text=True,
    )
    report = json.loads(proc.stdout.strip())
    assert proc.returncode == 0 and report["ok"] is True
    assert Path(report["output"]).is_file()
    assert not Path(str(out)).is_dir()
