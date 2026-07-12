# Standard libs
from pathlib import Path
from uuid import uuid4

# 3rdparty libs
from PIL.Image import Image

# Internal libs
from schema import Audio, Citation, Embedding, Metadata, SearchResult
from utils import pdftools


def make_ids(items: list) -> list[str]:
    return [str(uuid4()) for _ in items]


def make_texts(results: list[SearchResult]) -> list[str]:
    return [
        result.payload["text"]
        for result in results
        if result.payload.get("source") == "text"
    ]


def make_images(results: list[SearchResult]) -> list[Image]:
    return [
        pdftools.convert_pdf_page_to_pil_image(
            file_path=Path(result.payload["path"]),
            page_num=result.payload["page"],
        )
        for result in results
        if result.payload.get("source") == "image"
    ]


def make_audios(results: list[SearchResult]) -> list[Audio]:
    return [
        Audio(path=Path(result.payload["path"]))
        for result in results
        if result.payload.get("source") == "audio"
    ]


def make_citations(results: list[SearchResult]) -> list[Citation]:
    citations = []
    for result in results:
        payload = result.payload
        source = payload.get("source", "unknown")
        page = payload.get("page")
        citations.append(
            Citation(
                source=source,
                score=result.score,
                page=page + 1 if page is not None else None,
                filename=payload.get("filename"),
            )
        )
    return citations


def make_text_metadatas(texts: list[str], path: Path) -> list[Metadata]:
    return [
        {
            "text": text,
            "page": page_id,
            "filename": path.name,
            "path": str(path),
            "source": "text",
        }
        for page_id, text in enumerate(texts)
    ]


def make_image_metadatas(embeddings: list[Embedding], path: Path) -> list[Metadata]:
    return [
        {
            "page": page_id,
            "filename": path.name,
            "path": str(path),
            "source": "image",
        }
        for page_id in range(len(embeddings))
    ]


def make_audio_metadatas(embeddings: list[Embedding], path: Path) -> list[Metadata]:
    return [
        {
            "filename": path.name,
            "path": str(path),
            "source": "audio",
        }
        for _ in range(len(embeddings))
    ]
