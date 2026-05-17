# Standard libs
from pathlib import Path

# 3rdparty libs
from PIL.Image import Image
from pymupdf import Document, Page, Pixmap


def convert_pdf_to_pil_images(pdf_file: Path) -> list[Image]:
    images: list[Image] = []

    with Document(filename=pdf_file) as pdf:
        for page_id in range(pdf.page_count):
            page: Page = pdf.load_page(page_id)
            pixmap: Pixmap = page.get_pixmap()
            image: Image = Pixmap.pil_image(pixmap)
            images.append(image)

    return images


def convert_pdf_page_to_pil_image(pdf_file: Path, page_num: int) -> Image:
    with Document(filename=pdf_file) as pdf:
        page: Page = pdf.load_page(page_num)
        pixmap: Pixmap = page.get_pixmap()
        return Pixmap.pil_image(pixmap)
