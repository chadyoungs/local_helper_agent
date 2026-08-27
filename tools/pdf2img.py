import argparse
from pathlib import Path

from pdf2image import convert_from_path

# Convert the PDF pages to a list of images
# If on Windows and Poppler is not in PATH, you must specify the poppler_path argument
password_dict = {
    "2025": "205433",
    "2024": "000023",
    "2023": "135432",
    "2022": "355144",
    "2021": "413304",
    "2020": "312315",
    "2019": "001051",
    "2018": "441151",
    "2017": "412441",
    "2016": "413450",
}


def pdf2img(pdf_file_list, output_folder, passwords):
    for idx, pdf_file in enumerate(pdf_file_list):

        userpw = None if passwords is None else passwords[idx]

        images = convert_from_path(
            pdf_file,
            userpw=userpw,
            fmt="jpeg",  # Optional: specify output format like "jpeg" or "png"
            # poppler_path=r'C:\path\to\poppler-xx\bin' # Uncomment and edit this line for Windows if needed
        )

        # Iterate over the images list and save each page as a file
        for i, image in enumerate(images):
            # Filenames will include original name and page number
            image.save(f"{output_folder}/{Path(pdf_file).stem}_page_{i+1}.jpg", "JPEG")

    print(f"Conversion successful. Images saved in the '{output_folder}' directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--passwords", nargs="+")
    args = parser.parse_args()

    pdf2img(args.inputs, args.output, args.passwords)
