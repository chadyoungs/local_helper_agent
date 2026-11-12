"""End-to-end tests for tools/signature_fingerprint_cutout.py.

A photo-like sheet is synthesised (uneven light, vignette, JPEG noise) with a
black signature and a red fingerprint, then the tool is run as a subprocess --
exactly how the harness runs it -- and the PNG is verified by reading back
pixels, not just by checking that a file appeared.
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
TOOL = PROJECT / "tools" / "signature_fingerprint_cutout.py"


def _run(*args):
    proc = subprocess.run(
        [sys.executable, str(TOOL), *args], capture_output=True, text=True
    )
    report = json.loads(proc.stdout.strip()) if proc.stdout.strip() else {}
    return proc.returncode, report, proc.stderr


def _sheet(path: Path, size=(900, 620), vignette=0.32, ink=True, red=True) -> None:
    """Write a JPEG that looks like a photographed sheet of paper."""
    rng = np.random.default_rng(3)
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    radius = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    light = 1.0 - vignette * np.clip(radius, 0, 1.6) ** 1.4
    base = np.array([240.0, 236.0, 226.0], dtype=np.float32)  # warm paper
    img = base[None, None, :] * light[..., None]
    img += rng.normal(0, 3.0, img.shape).astype(np.float32)
    canvas = np.clip(img, 0, 255).astype(np.uint8)

    if ink:
        cv2.polylines(
            canvas,
            [np.array([[120, 330], [260, 250], [370, 350], [470, 280]])],
            False,
            (32, 28, 36),
            4,
            cv2.LINE_AA,
        )
    if red:
        for i in range(6):
            cv2.ellipse(canvas, (660, 360), (26 + i * 9, 62 + i * 7), 0, 0, 360,
                        (196, 44, 46), 2, cv2.LINE_AA)
        cv2.ellipse(canvas, (660, 360), (20, 48), 0, 0, 360, (196, 44, 46), -1, cv2.LINE_AA)

    canvas = cv2.GaussianBlur(canvas, (3, 3), 0.6)
    Image.fromarray(canvas).save(str(path), "JPEG", quality=82)


@pytest.fixture()
def sheet(tmp_path) -> Path:
    p = tmp_path / "sheet.jpg"
    _sheet(p)
    return p


def test_cutout_keeps_ink_and_drops_paper(sheet, tmp_path):
    out = tmp_path / "cutout.png"
    code, report, stderr = _run("--input", str(sheet), "--output", str(out))
    assert code == 0, (report, stderr)
    assert report["ok"] is True
    assert Path(report["output"]).is_file()
    assert report["has_black_ink"] and report["has_red_ink"]

    rgba = np.asarray(Image.open(report["output"]).convert("RGBA"))
    h, w = rgba.shape[:2]

    # paper is gone at the edges
    assert all(int(px[3]) == 0 for px in (rgba[1, 1], rgba[1, w - 2],
                                          rgba[h - 2, 1], rgba[h - 2, w - 2]))

    solid = rgba[rgba[..., 3] == 255][:, :3].astype(np.int16)
    assert len(solid) > 0
    darkest = solid[solid.sum(axis=1).argmin()]
    assert darkest.max() < 110, darkest                   # black ink stays black
    reddest = solid[(solid[:, 0] - solid[:, 1]).argmax()]
    assert reddest[0] > 120 and reddest[0] - reddest[1] > 50, reddest  # not BGR-swapped

    # a real alpha ramp, not a 1-bit mask
    assert len(np.unique(rgba[..., 3])) > 20


def test_blank_sheet_reports_clean_json_error(tmp_path):
    blank = tmp_path / "blank.jpg"
    _sheet(blank, ink=False, red=False)
    code, report, stderr = _run("--input", str(blank), "--output", str(tmp_path / "blank.png"))
    assert code != 0
    assert report["ok"] is False and report["stage"] == "detect"
    assert "Traceback" not in stderr


def test_missing_input_reports_clean_json_error(tmp_path):
    code, report, stderr = _run("--input", str(tmp_path / "nope.jpg"),
                                "--output", str(tmp_path / "x.png"))
    assert code != 0
    assert report["ok"] is False and report["stage"] == "input"
    assert "Traceback" not in stderr


def test_output_path_contract(sheet, tmp_path):
    """File target keeps its name, folder target derives one, .jpg is forced to .png."""
    folder = tmp_path / "outdir"
    folder.mkdir()
    code, report, _ = _run("--input", str(sheet), "--output", str(folder))
    assert code == 0 and Path(report["output"]).parent == folder
    assert Path(report["output"]).name == "sheet_cutout.png"

    code, report, _ = _run("--input", str(sheet), "--output", str(tmp_path / "named"))
    assert code == 0 and report["output"].endswith("named.png")

    code, report, _ = _run("--input", str(sheet), "--output", str(tmp_path / "forced.jpg"))
    assert code == 0 and report["output"].endswith("forced.png") and "note" in report
