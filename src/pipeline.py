# Standard libs
from pathlib import Path

# 3rdparty libs
from PIL.Image import Image

# Internal libs
from llm.gemini import GeminiClient
from rag.base import BaseRAG
from rag.image import ImageRAG
from rag.text import TextRAG
from schema import Context, Query, Response, SearchResult
from utils import pdf2img


def _make_texts(results: list[SearchResult]) -> list[str]:
    return [
        result.payload["text"]
        for result in results
        if result.payload.get("source") == "text"
    ]


def _make_images(results: list[SearchResult]) -> list[Image]:
    return [
        pdf2img.convert_pdf_page_to_pil_image(
            pdf_file=Path(result.payload["pdf_path"]),
            page_num=result.payload["page"],
        )
        for result in results
        if result.payload.get("source") == "image"
    ]


class ZiRAG(BaseRAG):
    def __init__(
        self,
        text_rag: TextRAG,
        image_rag: ImageRAG,
        llm: GeminiClient,
    ) -> None:
        self.text_rag = text_rag
        self.image_rag = image_rag
        self.llm = llm

    def index(self, pdf_file: Path) -> None:
        self.text_rag.index(pdf_file)
        self.image_rag.index(pdf_file)

    def search(self, query: Query, n_results: int = 10) -> list[SearchResult]:
        text_results = self.text_rag.search(query, n_results)
        image_results = self.image_rag.search(query, n_results)
        return text_results + image_results

    def generate(self, query: Query, n_results: int = 3) -> Response:
        results = self.search(query, n_results)
        texts = _make_texts(results)
        images = _make_images(results)
        context = Context(query=query, texts=texts, images=images)
        return self.llm.generate(context)
