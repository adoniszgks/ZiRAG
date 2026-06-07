# Standard libs
from pathlib import Path

# Internal libs
from rag.aural import AuralRAG
from rag.generation.llm.gemini import GeminiLLM
from rag.textual import TextualRAG
from rag.visual import VisualRAG
from schema import Context, Query, Response, SearchMode, SearchResult
from utils.ragtools import make_audios, make_citations, make_images, make_texts


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

    def search(
        self,
        query: Query,
        n_results: int = 10,
        use_textual: bool = True,
        use_visual: bool = True,
        use_aural: bool = True,
    ) -> list[SearchResult]:
        results = []
        if self.textual_rag is not None and use_textual:
            results += self.textual_rag.search(SearchMode.SIM, query, n_results)
        if self.visual_rag is not None and use_visual:
            results += self.visual_rag.search(query, n_results)
        if self.aural_rag is not None and use_aural:
            results += self.aural_rag.search(query, n_results)
        return results

    def generate(
        self,
        query: Query,
        n_results: int = 3,
        language: str = "English",
        use_textual: bool = True,
        use_visual: bool = True,
        use_aural: bool = True,
    ) -> Response:
        results = self.search(
            query=query,
            n_results=n_results,
            use_textual=use_textual,
            use_visual=use_visual,
            use_aural=use_aural,
        )
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
        response = self.llm.generate(context)
        response.citations = make_citations(results)
        return response
