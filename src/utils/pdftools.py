# Standard libs
from pathlib import Path

# 3rdparty libs
from PIL.Image import Image
from pymupdf import Document, Pixmap


def convert_pdf_to_pil_images(file_path: Path) -> list[Image]:
    images = []

    with Document(file_path) as pdf:
        for page_id in range(pdf.page_count):
            page = pdf.load_page(page_id)
            pixmap = page.get_pixmap()
            images.append(Pixmap.pil_image(pixmap))

    return images


def convert_pdf_page_to_pil_image(file_path: Path, page_num: int) -> Image:
    with Document(file_path) as pdf:
        page = pdf.load_page(page_num)
        pixmap = page.get_pixmap()
        return Pixmap.pil_image(pixmap)


def extract_pdf_texts(file_path: Path) -> list[str]:
    with Document(file_path) as pdf:
        return [str(pdf.load_page(i).get_text()) for i in range(pdf.page_count)]
