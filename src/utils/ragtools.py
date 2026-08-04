# Standard libs
from pathlib import Path
from uuid import uuid4

# 3rdparty libs
from PIL.Image import Image

# Internal libs
from schema import Citation, Embedding, Metadata, Query, SearchResult
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


def make_description(result: SearchResult) -> dict:
    payload = result.payload
    source = payload.get("source", "unknown")
    page = payload.get("page")
    return {
        "source": source,
        "filename": payload.get("filename"),
        "page": page + 1 if page is not None else None,
        "score": result.score,
        "text": payload.get("text", "") if source == "text" else None,
    }


def make_log(
    query: Query,
    textual_results: list[SearchResult],
    visual_results: list[SearchResult],
    aural_results: list[SearchResult],
) -> dict:
    return {
        "query": {
            "texts": query.texts,
            "images": len(query.images) if query.images else 0,
            "audios": len(query.audios) if query.audios else 0,
        },
        "textual": [make_description(result) for result in textual_results],
        "visual": [make_description(result) for result in visual_results],
        "aural": [make_description(result) for result in aural_results],
    }


def make_text_metadatas(texts: list[str], path: Path) -> list[Metadata]:
    return [
        {
            "text": text,
            "page": page_num,
            "filename": path.name,
            "path": str(path),
            "source": "text",
        }
        for page_num, text in enumerate(texts)
    ]


def make_image_metadatas(embeddings: list[Embedding], path: Path) -> list[Metadata]:
    return [
        {
            "page": page_num,
            "filename": path.name,
            "path": str(path),
            "source": "image",
        }
        for page_num in range(len(embeddings))
    ]
