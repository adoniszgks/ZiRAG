# Standard libs
from pathlib import Path

# 3rdparty libs
from PIL.Image import Image
from pymupdf import Document, Page, Pixmap


def convert_pdf_to_pil_images(pdf_path: Path) -> list[Image]:
    images: list[Image] = []

    with Document(pdf_path) as pdf_file:
        for page_id in range(pdf_file.page_count):
            page: Page = pdf_file.load_page(page_id)
            pixmap: Pixmap = page.get_pixmap()
            image: Image = Pixmap.pil_image(pixmap)
            images.append(image)

    return images
