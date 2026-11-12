"""Change the background colour of an ID (portrait) photo.

Two paths, chosen automatically:

1. ``rembg`` — model based, best quality, needs the optional extra dependency.
2. **Solid-background fallback** — no model, no download: the background colour
   is sampled from the photo border and every region *connected to the border*
   that matches it is made transparent. This keeps the tool working on a plain
   Python 3.9 install where ``rembg``/``onnxruntime`` wheels are unavailable.

Output contract (shared with the harness and the other tools): exactly one JSON
object on stdout, for success and for failure; no traceback on stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_SUFFIX = ".jpg"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Change ID photo background colour.")
    ap.add_argument("--input", required=True, help="Input photo (absolute path).")
    ap.add_argument("--output", required=True,
                    help="Output image file, or a folder to write <input_stem>_bg_changed.jpg into.")
    ap.add_argument("--target-color", required=True,
                    help="Target background colour, hex, e.g. #FF0000 / #0000FF / #FFFFFF.")
    ap.add_argument("--tolerance", type=int, default=45,
                    help="Fallback mode: max per-channel difference from the border colour "
                         "that still counts as background (0-255). Default 45.")
    return ap.parse_args(argv)


def resolve_output_path(raw: str, input_path: Path) -> Path:
    """File target keeps its name; folder target gets a derived name."""
    p = Path(raw)
    if p.exists() and p.is_dir():
        return p / ("%s_bg_changed%s" % (input_path.stem, DEFAULT_SUFFIX))
    suffix = p.suffix.lower()
    if suffix in IMAGE_EXTS:
        return p
    if suffix == "":
        return Path(str(p) + DEFAULT_SUFFIX)
    return p / ("%s_bg_changed%s" % (input_path.stem, DEFAULT_SUFFIX))


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    value = hex_color.strip().lstrip("#")
    if len(value) != 6:
        raise ValueError("target colour must be a 6-digit hex code like #FF0000")
    return tuple(int(value[i: i + 2], 16) for i in (0, 2, 4))


def _remove_with_rembg(img: Image.Image) -> Image.Image:
    import rembg  # heavy optional dependency, imported only when available

    return rembg.remove(
        img, alpha_matting=True, alpha_matting_foreground_threshold=240
    ).convert("RGBA")


def _remove_with_solid_colour(img: Image.Image, tolerance: int) -> Image.Image:
    """Model-free removal for a plain studio background.

    Only regions that *touch the image border* and match the border colour are
    dropped, so a subject wearing the same colour as the backdrop survives.
    """
    rgb = np.asarray(img.convert("RGB")).astype(np.int16)
    border = np.concatenate([rgb[0, :], rgb[-1, :], rgb[:, 0], rgb[:, -1]], axis=0)
    background = np.median(border, axis=0)
    diff = np.abs(rgb - background).max(axis=2)
    mask = (diff <= tolerance).astype(np.uint8)

    count, labels = cv2.connectedComponents(mask, connectivity=4)
    edge_labels = set(
        np.unique(
            np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
        ).tolist()
    ) - {0}
    keep = np.ones(count, dtype=np.uint8)
    if edge_labels:
        keep[list(edge_labels)] = 0
    alpha = (keep[labels] * 255).astype(np.uint8)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    return Image.fromarray(np.dstack([rgb.astype(np.uint8), alpha]))


def change_background(
    input_path: Path, target_rgb: Tuple[int, int, int], tolerance: int
) -> Tuple[Image.Image, str, Optional[str]]:
    """Return (composited RGB image, method name, fallback reason)."""
    img = Image.open(input_path).convert("RGBA")

    foreground = None
    method = "rembg"
    reason = None
    try:
        foreground = _remove_with_rembg(img)
    except Exception as exc:  # noqa: BLE001 - missing model / no wheel / runtime error
        reason = "%s: %s" % (type(exc).__name__, exc)
        method = "solid-background-fallback"
        foreground = _remove_with_solid_colour(img, tolerance)

    background = Image.new("RGBA", foreground.size, target_rgb + (255,))
    final = Image.alpha_composite(background, foreground)
    return final.convert("RGB"), method, reason


def run(args: argparse.Namespace) -> Dict[str, Any]:
    inp = Path(args.input)
    if not inp.is_file():
        return {"ok": False, "stage": "input", "error": "input file not found: %s" % inp}

    try:
        target_rgb = hex_to_rgb(args.target_color)
    except ValueError as exc:
        return {"ok": False, "stage": "params", "error": str(exc)}

    try:
        final, method, reason = change_background(inp, target_rgb, args.tolerance)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "stage": "process", "error": "%s: %s" % (type(exc).__name__, exc)}

    out = resolve_output_path(args.output, inp)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".png":
        final.save(str(out), "PNG")
    else:
        final.save(str(out), "JPEG", quality=95)

    report: Dict[str, Any] = {
        "ok": True,
        "output": str(out),
        "method": method,
        "target_color": "#%02X%02X%02X" % target_rgb,
        "size": {"width": final.size[0], "height": final.size[1]},
    }
    if reason:
        report["fallback_reason"] = reason
    return report


def main(argv: Optional[List[str]] = None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2

    try:
        report = run(args)
    except Exception as exc:  # noqa: BLE001 - the JSON contract must always hold
        report = {"ok": False, "stage": "unexpected", "error": "%s: %s" % (type(exc).__name__, exc)}

    print(json.dumps(report, ensure_ascii=False))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
