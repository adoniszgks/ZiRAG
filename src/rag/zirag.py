# Standard libs
from pathlib import Path

# 3rdparty libs
from PIL.Image import Image

# Internal libs
from rag.aural import AuralRAG
from rag.base import BaseRAG
from rag.generation.llm.gemini import GeminiLLM
from rag.textual import TextualRAG
from rag.visual import VisualRAG
from schema import Audio, Context, Query, Response, SearchMode, SearchResult
from utils import pdftools


def _make_texts(results: list[SearchResult]) -> list[str]:
    return [
        result.payload["text"]
        for result in results
        if result.payload.get("source") == "text"
    ]


def _make_images(results: list[SearchResult]) -> list[Image]:
    return [
        pdftools.convert_pdf_page_to_pil_image(
            pdf_file=Path(result.payload["pdf_path"]),
            page_num=result.payload["page"],
        )
        for result in results
        if result.payload.get("source") == "image"
    ]


def _make_audios(results: list[SearchResult]) -> list[Audio]:
    return [
        Audio(path=Path(result.payload["audio_path"]))
        for result in results
        if result.payload.get("source") == "audio"
    ]


class ZiRAG(BaseRAG):
    def __init__(
        self,
        textual_rag: TextualRAG,
        visual_rag: VisualRAG,
        aural_rag: AuralRAG,
        llm: GeminiLLM,
    ) -> None:
        self.textual_rag = textual_rag
        self.visual_rag = visual_rag
        self.aural_rag = aural_rag
        self.llm = llm

    def index(self, pdf_file: Path, audio_file: Path | None = None) -> None:
        self.textual_rag.index(pdf_file)
        self.visual_rag.index(pdf_file)
        self.aural_rag.index(audio_file)

    def search(self, query: Query, n_results: int = 10) -> list[SearchResult]:
        text_results = self.textual_rag.search(SearchMode.SIM, query, n_results)
        image_results = self.visual_rag.search(query, n_results)
        audio_results = self.aural_rag.search(query, n_results)
        return text_results + image_results + audio_results

    def generate(self, query: Query, n_results: int = 3) -> Response:
        results = self.search(query, n_results)
        texts = _make_texts(results)
        images = _make_images(results)
        audios = _make_audios(results)
        context = Context(query=query, texts=texts, images=images, audios=audios)
        return self.llm.generate(context)
