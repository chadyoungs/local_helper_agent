import argparse
import json
import re
from pathlib import Path

from pypdf import PaperSize, PdfReader, PdfWriter


def main():
    parser = argparse.ArgumentParser(
        description="PDF merge tool, force all pages to A4 portrait, support built‑in blank pages between documents"
    )
    parser.add_argument(
        "--inputs",
        required=True,
        type=str,
        help="Comma‑separated list of logical pdf filenames to merge",
    )
    parser.add_argument(
        "--blank-count",
        type=int,
        default=0,
        help="Number of blank pages to insert between documents, 0 means no blank pages",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Logical filename for merged output pdf",
    )

    args = parser.parse_args()
    ret = {"ok": False, "error": None, "input_files": [], "output_file": None}

    # Target dimensions for standard A4 in points (595.27 x 841.89)
    target_width = float(PaperSize.A4.width)
    target_height = float(PaperSize.A4.height)

    try:
        # Locate built‑in blank.pdf static file
        blank_pdf = Path(__file__).parent.parent / "statics" / "blank.pdf"
        if not blank_pdf.exists() or not blank_pdf.is_file():
            raise FileNotFoundError(
                f"Built‑in blank pdf not found at: {blank_pdf.resolve()}"
            )
        
        try:
            input_paths = [f.strip() for f in re.split(r"[,\s]+", args.inputs) if f.strip()]
        except:
            input_paths = args.inputs

        ret["input_files"] = input_paths

        writer = PdfWriter()

        for idx, pdf_path in enumerate(input_paths):
            reader = PdfReader(str(pdf_path))
            for src_page in reader.pages:
                src_page.scale_to(width=target_width, height=target_height)
                writer.add_page(src_page)

            # Insert blank pages between documents, skip after the last document
            blank_count = int(args.blank_count)
            if blank_count > 0 and idx != len(input_paths) - 1:
                blank_reader = PdfReader(str(blank_pdf))
                for _ in range(blank_count):
                    writer.add_page(blank_reader.pages[0])

        out_path = Path(args.output)
        with open(out_path, "wb") as f_out:
            writer.write(f_out)

        ret["ok"] = True
        ret["output_file"] = str(out_path)

    except Exception as e:
        ret["error"] = str(e)

    print(json.dumps(ret, ensure_ascii=False))


if __name__ == "__main__":
    main()
