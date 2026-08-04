# Standard libs
from pathlib import Path

# Internal libs
from rag.base import BaseRAG
from rag.generation.llm.gemini import GeminiLLM
from rag.retrieval.models.embedder.audio_embedder import AudioEmbedder
from schema import Context, Query, Response, SearchResult
from utils.pdftools import extract_pdf_texts
from utils.ragtools import make_ids, make_text_metadatas, make_texts
from vectorstore.base import BaseIndexer


class AuralRAG(BaseRAG):
    def __init__(
        self,
        indexer: BaseIndexer,
        embedder: AudioEmbedder,
        llm: GeminiLLM,
    ) -> None:
        self.indexer = indexer
        self.embedder = embedder
        self.llm = llm

    def index(self, file_path: Path | None) -> None:
        if not file_path:
            return
        texts = extract_pdf_texts(file_path)
        embeddings = self.embedder.embed_text(texts)
        ids = make_ids(embeddings)
        metadatas = make_text_metadatas(texts, file_path)
        self.indexer.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def search(self, query: Query, n_results: int = 5) -> list[SearchResult]:
        if not query.audios:
            return []
        paths = [str(audio.path) for audio in query.audios]
        query_embeddings = self.embedder.embed_audio(paths)
        return self.indexer.search(query_embeddings, n_results)

    def generate(self, query: Query, n_results: int = 3) -> Response:
        results = self.search(query, n_results)
        context = Context(query=query, texts=make_texts(results))
        return self.llm.generate(context)
