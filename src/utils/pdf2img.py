# Standard libs
from pathlib import Path

# 3rdparty libs
from PIL.Image import Image
from pymupdf import Document, Pixmap


def convert_pdf_to_pil_images(pdf_file: Path) -> list[Image]:
    images = []

    with Document(pdf_file) as pdf:
        for page_id in range(pdf.page_count):
            page = pdf.load_page(page_id)
            pixmap = page.get_pixmap()
            images.append(Pixmap.pil_image(pixmap))

    return images


def convert_pdf_page_to_pil_image(pdf_file: Path, page_num: int) -> Image:
    with Document(pdf_file) as pdf:
        page = pdf.load_page(page_num)
        pixmap = page.get_pixmap()
        return Pixmap.pil_image(pixmap)
