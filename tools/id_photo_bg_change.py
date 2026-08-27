import argparse
from pathlib import Path

import rembg
from PIL import Image

# ------------------------------
# ID Photo Background Changer
# Replace the background of an ID photo with a user‑specified solid color(hex).
# No auto‑swap / auto‑detect background color logic.
# ------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Change ID photo background color.")
    parser.add_argument(
        "--input", required=True, help="Input image filename in workspace/input"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory in workspace/output"
    )
    parser.add_argument(
        "--target-color",
        required=True,
        help="Target background color in hex format, e.g. #FF0000(red), #0000FF(blue), #FFFFFF(white)",
    )
    return parser.parse_args()


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def change_background(input_path: str, output_dir: Path, target_color_hex: str):
    """
    Remove original background and replace it with solid target color.
    """
    img = Image.open(input_path).convert("RGBA")

    # remove background with rembg
    output = rembg.remove(
        img, alpha_matting=True, alpha_matting_foreground_threshold=240
    )
    foreground = output.convert("RGBA")

    target_rgb = hex_to_rgb(target_color_hex)
    background = Image.new("RGBA", foreground.size, target_rgb + (255,))

    final = Image.alpha_composite(background, foreground)
    final = final.convert("RGB")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"{Path(input_path).stem}_bg_changed.jpg"
    save_path = output_dir / output_filename

    final.save(str(save_path), "JPEG", quality=200)
    print(f"Background changed successfully: {save_path}")


def main():
    args = parse_args()
    input_file = Path(args.input)
    out_dir = Path(args.output)
    target_hex = args.target_color

    if target_hex is None or str(target_hex).strip() == "":
        raise ValueError(
            "target‑color cannot be None or empty. Must provide hex color like #FF0000"
        )
    target_hex = target_hex.strip()

    change_background(str(input_file), out_dir, target_hex)


if __name__ == "__main__":
    main()
