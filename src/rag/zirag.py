# Standard libs
from pathlib import Path

# Internal libs
from rag.aural import AuralRAG
from rag.base import BaseRAG
from rag.generation.llm.gemini import GeminiLLM
from rag.textual import TextualRAG
from rag.visual import VisualRAG
from schema import Context, Query, Response, SearchMode, SearchResult
from utils.ragtools import make_audios, make_images, make_texts


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

    def index(self, text: Path, image: Path, audio: Path) -> None:
        self.textual_rag.index(text)
        self.visual_rag.index(image)
        self.aural_rag.index(audio)

    def search(self, query: Query, n_results: int = 10) -> list[SearchResult]:
        text_results = self.textual_rag.search(SearchMode.SIM, query, n_results)
        image_results = self.visual_rag.search(query, n_results)
        audio_results = self.aural_rag.search(query, n_results)
        return text_results + image_results + audio_results

    def generate(self, query: Query, n_results: int = 3) -> Response:
        results = self.search(query, n_results)
        texts = make_texts(results)
        images = make_images(results)
        audios = make_audios(results)
        context = Context(query=query, texts=texts, images=images, audios=audios)
        return self.llm.generate(context)
