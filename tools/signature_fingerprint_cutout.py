#!/usr/bin/env python3
"""Cut a signature + fingerprint out of a photographed sheet.

The paper background (grey, yellowed, unevenly lit, often vignetted) is removed
and only the ink is kept:

* black ink  -> detected as *relative darkness* against a locally estimated
  paper brightness, so a dark corner of the photo is not mistaken for ink.
* red ink    -> detected by HSV **saturation** plus a red hue gate, which keeps
  faint fingerprints that a plain "R > G,B" rule would drop.

Output is an RGBA PNG whose background is fully transparent; the ink keeps its
original colour (black stays black, red stays red).

Contract (shared with the harness and the other tools in this project):

* ``--input``  absolute path of an existing image (rewritten by the harness)
* ``--output`` absolute path; used as a file when it carries an image suffix,
  as a folder otherwise
* stdout carries **exactly one JSON object** -- success *and* failure reports,
  so the model always has something parseable to read. Nothing is written to
  stderr and no traceback ever escapes.

Exit code 0 on success, non-zero on failure (still with a JSON body on stdout).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_SUFFIX = ".png"
# Photos up to this size are handled at full resolution; larger ones are scaled
# down first so the filters stay fast and memory bounded.
MAX_WORK_SIDE = 2600


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Remove paper background, keep black signature + red fingerprint."
    )
    ap.add_argument("--input", required=True, help="Input photo (absolute path).")
    ap.add_argument(
        "--output",
        required=True,
        help="Output PNG file, or a folder to write <input_stem>_cutout.png into.",
    )
    ap.add_argument(
        "--ink-min",
        type=float,
        default=8.0,
        help="Darkness (percent) where ink starts to appear. Default 8.",
    )
    ap.add_argument(
        "--ink-full",
        type=float,
        default=55.0,
        help="Darkness (percent) where ink is fully opaque. Default 55.",
    )
    ap.add_argument(
        "--red-min",
        type=float,
        default=18.0,
        help="Saturation (percent) where red ink starts to appear. Default 18.",
    )
    ap.add_argument(
        "--red-full",
        type=float,
        default=45.0,
        help="Saturation (percent) where red ink is fully opaque. Default 45.",
    )
    ap.add_argument(
        "--cutoff",
        type=float,
        default=4.0,
        help="Drop alpha below this percent to clean faint paper residue. Default 4.",
    )
    ap.add_argument(
        "--feather",
        type=int,
        default=1,
        help="Edge smoothing radius in pixels, 0 disables it. Default 1.",
    )
    ap.add_argument(
        "--padding",
        type=int,
        default=8,
        help="Transparent padding kept around the trimmed ink. Default 8.",
    )
    ap.add_argument(
        "--min-area",
        type=float,
        default=0.05,
        help="Speck removal: drop blobs smaller than this percent of the image "
             "area (0 disables). Default 0.05.",
    )
    ap.add_argument(
        "--no-fill",
        action="store_true",
        help="Do not fill the inner gaps of stamps/fingerprints (keeps ridges hollow).",
    )
    return ap.parse_args(argv)


# --------------------------------------------------------------------------- #
# Output path contract
# --------------------------------------------------------------------------- #
def resolve_output_path(raw: str, input_path: Path) -> Tuple[Path, Optional[str]]:
    """Return (path, note). Never create a folder where a file was expected."""
    p = Path(raw)
    if p.exists() and p.is_dir():
        # The harness pre-creates a folder when the logical name had no dot.
        return p / f"{input_path.stem}_cutout{DEFAULT_SUFFIX}", None
    suffix = p.suffix.lower()
    if suffix == "":
        return Path(str(p) + DEFAULT_SUFFIX), None
    if suffix in IMAGE_EXTS:
        if suffix != DEFAULT_SUFFIX:
            # A transparent result cannot be a JPEG; force PNG and tell the caller.
            return p.with_suffix(DEFAULT_SUFFIX), (
                "output extension forced to .png (transparency needs an alpha channel)"
            )
        return p, None
    # no image suffix at all -> treat it as a folder
    return p / f"{input_path.stem}_cutout{DEFAULT_SUFFIX}", None


# --------------------------------------------------------------------------- #
# Image loading
# --------------------------------------------------------------------------- #
def load_rgb(path: Path) -> np.ndarray:
    """Read an image as RGB float32, honouring EXIF orientation and scaling down."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        elif im.mode == "RGBA":
            # flatten transparency onto white so the paper estimate is sane
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        rgb = np.asarray(im.convert("RGB"))
    arr = rgb.astype(np.float32)
    longest = max(arr.shape[0], arr.shape[1])
    if longest > MAX_WORK_SIDE:
        scale = MAX_WORK_SIDE / float(longest)
        new_size = (max(1, int(arr.shape[1] * scale)), max(1, int(arr.shape[0] * scale)))
        arr = cv2.resize(arr, new_size, interpolation=cv2.INTER_AREA)
    return arr


# --------------------------------------------------------------------------- #
# Alpha estimation
# --------------------------------------------------------------------------- #
def estimate_paper(brightness: np.ndarray) -> np.ndarray:
    """Estimate the local paper brightness with a large median filter.

    The photo may be bright in the middle and dark in the corners; a global
    threshold would carve the vignette out as "ink". A median estimate of the
    paper level follows the illumination gradient instead.
    """
    h, w = brightness.shape[:2]
    # Downscale, so a large median kernel stays cheap even for big photos.
    target = 320
    scale = min(1.0, target / float(max(h, w)))
    small = cv2.resize(
        brightness, (max(3, int(w * scale)), max(3, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    u8 = np.clip(small, 0, 255).astype(np.uint8)
    k = max(5, (min(u8.shape[:2]) // 8) | 1)
    if k % 2 == 0:
        k += 1
    # medianBlur needs an odd kernel no larger than the image itself
    k = min(k, u8.shape[0] - 1 if u8.shape[0] % 2 == 0 else u8.shape[0],
            u8.shape[1] - 1 if u8.shape[1] % 2 == 0 else u8.shape[1])
    if k % 2 == 0:
        k -= 1
    k = max(3, k)
    paper_small = cv2.medianBlur(u8, k)
    paper = cv2.resize(paper_small, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.maximum(paper.astype(np.float32), 1.0)


def ramp(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Smooth 0..1 ramp between ``lo`` and ``hi``; tolerates a tiny span."""
    span = max(1e-3, float(hi) - float(lo))
    return np.clip((values - float(lo)) / span, 0.0, 1.0)


def ink_alpha(
    rgb: np.ndarray,
    ink_min: float,
    ink_full: float,
    red_min: float,
    red_full: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (alpha, dark_component, red_component) as float32 arrays in 0..1."""
    brightness = rgb.max(axis=2)
    paper = estimate_paper(brightness)
    darkness = 1.0 - (brightness / paper)
    dark = ramp(darkness, ink_min / 100.0, ink_full / 100.0)

    # HSV via an explicit BGR hop so channel order is unambiguous.
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.float32)   # 0..179 in OpenCV
    sat = hsv[..., 1].astype(np.float32) / 255.0
    red_hue = ((hue <= 20) | (hue >= 150)).astype(np.float32)
    red = ramp(sat, red_min / 100.0, red_full / 100.0) * red_hue

    alpha = np.maximum(dark, red)
    return alpha.astype(np.float32), dark, red


def _representative_colour(rgb: np.ndarray, mask: np.ndarray) -> List[float]:
    """Median colour of the strong-ink pixels of one component (black or red)."""
    px = rgb[mask]
    if px.size == 0:
        return [0.0, 0.0, 0.0]
    return [float(v) for v in np.median(px, axis=0)]


def solidify(rgb: np.ndarray, dark: np.ndarray, red: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fill paper gaps inside a stamp/fingerprint with the ink's own colour.

    A fingerprint is a set of ridges with paper in between; keeping those gaps
    transparent makes it look torn. We close each ink component and fill its
    outer contour, but paint the newly covered pixels with the *median ink
    colour* -- using the paper colour would leave a pale opaque blob.

    Returns ``(rgb_with_filled_colour, extra_alpha)``.
    """
    out_rgb = rgb.astype(np.float32).copy()
    extra = np.zeros(rgb.shape[:2], dtype=np.float32)
    for component in (dark, red):
        mask = component > 0.5
        if not mask.any():
            continue
        h, w = mask.shape[:2]
        k = max(3, int(min(h, w) * 0.012) | 1)
        closed = cv2.morphologyEx(
            mask.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
        )
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        filled = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
        add = filled.astype(bool) & ~mask  # only the newly covered paper gaps
        if not add.any():
            continue
        colour = _representative_colour(rgb, mask)
        out_rgb[add] = colour
        extra[add] = 1.0
    return out_rgb, extra


def drop_specks(alpha: np.ndarray, min_area_pct: float) -> np.ndarray:
    """Remove isolated JPEG-noise blobs smaller than ``min_area_pct`` of the image."""
    if min_area_pct <= 0:
        return alpha
    mask = (alpha > 0.0).astype(np.uint8)
    if not mask.any():
        return alpha
    total = mask.shape[0] * mask.shape[1]
    min_area = max(1, int(total * min_area_pct / 100.0))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(count, dtype=np.uint8)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            keep[label] = 1
    return alpha * keep[labels]


def trim_to_content(alpha: np.ndarray, padding: int) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop to the ink bounding box, using a strong threshold so noise cannot win."""
    ys, xs = np.where(alpha > (50.0 / 255.0))
    if ys.size == 0:
        return alpha, (0, 0, alpha.shape[1], alpha.shape[0])
    h, w = alpha.shape[:2]
    y0 = max(0, int(ys.min()) - padding)
    x0 = max(0, int(xs.min()) - padding)
    y1 = min(h, int(ys.max()) + 1 + padding)
    x1 = min(w, int(xs.max()) + 1 + padding)
    return alpha[y0:y1, x0:x1], (x0, y0, x1, y1)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> Dict[str, Any]:
    inp = Path(args.input)
    if not inp.is_file():
        return {"ok": False, "stage": "input", "error": "input file not found: %s" % inp}

    try:
        rgb = load_rgb(inp)
    except Exception as exc:  # noqa: BLE001 - reported as JSON, never a traceback
        return {"ok": False, "stage": "decode", "error": "cannot read image: %s" % exc}

    if args.ink_full <= args.ink_min:
        return {
            "ok": False,
            "stage": "params",
            "error": "--ink-full must be greater than --ink-min",
        }
    if args.red_full <= args.red_min:
        return {
            "ok": False,
            "stage": "params",
            "error": "--red-full must be greater than --red-min",
        }

    alpha, dark, red = ink_alpha(
        rgb,
        ink_min=args.ink_min,
        ink_full=args.ink_full,
        red_min=args.red_min,
        red_full=args.red_full,
    )

    if not args.no_fill:
        rgb, filled_alpha = solidify(rgb, dark, red)
        alpha = np.maximum(alpha, filled_alpha)
    if args.feather > 0:
        k = 2 * args.feather + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    if args.cutoff > 0:
        alpha[alpha < (args.cutoff / 100.0)] = 0.0
    alpha = drop_specks(alpha, args.min_area)

    ink_pixels = int((alpha > 0).sum())
    if ink_pixels == 0:
        return {
            "ok": False,
            "stage": "detect",
            "error": (
                "no ink detected; try raising --ink-full / lowering --ink-min "
                "if the photo is dark, or --red-min if the stamp is faint"
            ),
            "image": {"width": int(rgb.shape[1]), "height": int(rgb.shape[0])},
        }

    # Trim in the full-resolution coordinate space, then crop colour and alpha
    # with the same box so the two arrays keep matching dimensions.
    alpha_crop, box = trim_to_content(alpha, args.padding)
    x0, y0, x1, y1 = box
    rgb_crop = rgb[y0:y1, x0:x1]

    out, note = resolve_output_path(args.output, inp)
    out.parent.mkdir(parents=True, exist_ok=True)

    rgba = np.dstack(
        [
            rgb_crop.astype(np.uint8),
            np.clip(alpha_crop * 255.0, 0, 255).astype(np.uint8),
        ]
    )
    # Saving with PIL keeps RGBA semantics; cv2 would silently read RGBA as BGRA
    # and swap red/blue in the red fingerprint.
    Image.fromarray(rgba).save(str(out), "PNG")

    h, w = rgba.shape[:2]
    ink_ratio = float((rgba[..., 3] > 0).sum()) / float(h * w) if h * w else 0.0
    report: Dict[str, Any] = {
        "ok": True,
        "output": str(out),
        "image": {
            "width": int(w),
            "height": int(h),
            "source_width": int(rgb.shape[1]),
            "source_height": int(rgb.shape[0]),
            "crop_box": [int(v) for v in box],
        },
        "ink_pixels": int((rgba[..., 3] > 0).sum()),
        "ink_ratio": round(ink_ratio, 5),
        "has_black_ink": bool((dark > 0.5).any()),
        "has_red_ink": bool((red > 0.5).any()),
        "params": {
            "ink_min": args.ink_min,
            "ink_full": args.ink_full,
            "red_min": args.red_min,
            "red_full": args.red_full,
            "cutoff": args.cutoff,
            "feather": args.feather,
            "fill_gaps": not args.no_fill,
        },
    }
    if note:
        report["note"] = note
    return report


def main(argv: Optional[List[str]] = None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:  # argparse already printed its own help
        return int(exc.code) if isinstance(exc.code, int) else 2

    try:
        report = run(args)
    except Exception as exc:  # noqa: BLE001 - the JSON contract must always hold
        report = {"ok": False, "stage": "unexpected", "error": "%s: %s" % (type(exc).__name__, exc)}

    print(json.dumps(report, ensure_ascii=False))
    if report.get("ok"):
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
