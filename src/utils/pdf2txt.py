# Standard libs
from pathlib import Path

# 3rdparty libs
from pymupdf import Document


def extract_pdf_texts(pdf_file: Path) -> list[str]:
    with Document(pdf_file) as pdf:
        return [str(pdf.load_page(i).get_text()) for i in range(pdf.page_count)]
