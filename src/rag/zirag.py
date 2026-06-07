# Standard libs
from pathlib import Path

# Internal libs
from rag.aural import AuralRAG
from rag.generation.llm.gemini import GeminiLLM
from rag.textual import TextualRAG
from rag.visual import VisualRAG
from schema import Context, Query, Response, SearchMode, SearchResult
from utils.ragtools import make_audios, make_images, make_texts


class ZiRAG:
    def __init__(
        self,
        textual_rag: TextualRAG | None = None,
        visual_rag: VisualRAG | None = None,
        aural_rag: AuralRAG | None = None,
        llm: GeminiLLM | None = None,
    ) -> None:
        self.textual_rag = textual_rag
        self.visual_rag = visual_rag
        self.aural_rag = aural_rag
        self.llm = llm

    def clear(self) -> None:
        if self.textual_rag is not None:
            self.textual_rag.indexer.clear()
        if self.visual_rag is not None:
            self.visual_rag.indexer.clear()
        if self.aural_rag is not None:
            self.aural_rag.indexer.clear()

    def index(self, text: Path, image: Path, audio: Path) -> None:
        if self.textual_rag is not None:
            self.textual_rag.index(text)
        if self.visual_rag is not None:
            self.visual_rag.index(image)
        if self.aural_rag is not None:
            self.aural_rag.index(audio)

    def search(self, query: Query, n_results: int = 10) -> list[SearchResult]:
        texts_results, image_results, audio_results = [], [], []

        if self.textual_rag is not None:
            texts_results = self.textual_rag.search(SearchMode.SIM, query, n_results)
        if self.visual_rag is not None:
            image_results = self.visual_rag.search(query, n_results)
        if self.aural_rag is not None:
            audio_results = self.aural_rag.search(query, n_results)

        return texts_results + image_results + audio_results

    def generate(
        self,
        query: Query,
        n_results: int = 3,
        language: str = "English",
    ) -> Response:
        results = self.search(query, n_results)
        texts = make_texts(results)
        images = make_images(results)
        audios = make_audios(results)
        context = Context(
            query=query,
            texts=texts,
            images=images,
            audios=audios,
            language=language,
        )
        return self.llm.generate(context)
