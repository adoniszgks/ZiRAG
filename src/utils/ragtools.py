# Standard libs
from pathlib import Path
from uuid import uuid4

# Internal libs
from schema import Embedding, Metadata


def make_ids(items: list) -> list[str]:
    return [str(uuid4()) for _ in items]


def make_text_metadatas(texts: list[str], path: Path) -> list[Metadata]:
    return [
        {
            "page": page_id,
            "pdf": path.name,
            "pdf_path": str(path),
            "source": "text",
            "text": text,
        }
        for page_id, text in enumerate(texts)
    ]


def make_image_metadatas(embeddings: list[Embedding], path: Path) -> list[Metadata]:
    return [
        {
            "page": page_id,
            "pdf": path.name,
            "pdf_path": str(path),
            "source": "image",
        }
        for page_id in range(len(embeddings))
    ]


def make_audio_metadatas(embeddings: list[Embedding], path: Path) -> list[Metadata]:
    return [
        {
            "audio": path.name,
            "audio_path": str(path),
            "source": "audio",
        }
        for _ in range(len(embeddings))
    ]
