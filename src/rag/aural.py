# Standard libs
from pathlib import Path

# Internal libs
from rag.base import BaseRAG
from rag.retrieval.models.embedder.audio_embedder import AudioEmbedder
from schema import Query, SearchResult
from utils.pdftools import extract_pdf_texts
from utils.ragtools import make_ids, make_text_metadatas
from vectorstore.base import BaseIndexer


class AuralRAG(BaseRAG):
    def __init__(
        self,
        indexer: BaseIndexer,
        embedder: AudioEmbedder,
    ) -> None:
        self.indexer = indexer
        self.embedder = embedder

    def index(self, file_path: Path | None) -> None:
        if not file_path:
            return
        texts = extract_pdf_texts(file_path)
        embeddings = self.embedder.embed_text(texts)
        ids = make_ids(embeddings)
        metadatas = make_text_metadatas(texts, file_path)
        self.indexer.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def search(self, query: Query, n_results: int) -> list[SearchResult]:
        if not query.audios:
            return []
        paths = [str(audio.path) for audio in query.audios]
        query_embeddings = self.embedder.embed_audio(paths)
        return self.indexer.search(query_embeddings, n_results)
