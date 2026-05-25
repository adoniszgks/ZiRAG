# Standard libs
from pathlib import Path
from uuid import uuid4

# Internal libs
from schema import Embedding, Metadata


def make_ids(items: list) -> list[str]:
    return [str(uuid4()) for _ in items]


def make_text_metadatas(texts: list[str], pdf_file: Path) -> list[Metadata]:
    return [
        {
            "page": page_id,
            "pdf": pdf_file.name,
            "pdf_path": str(pdf_file),
            "source": "text",
            "text": text,
        }
        for page_id, text in enumerate(texts)
    ]


def make_image_metadatas(embeddings: list[Embedding], pdf_file: Path) -> list[Metadata]:
    return [
        {
            "page": page_id,
            "pdf": pdf_file.name,
            "pdf_path": str(pdf_file),
            "source": "image",
        }
        for page_id in range(len(embeddings))
    ]
