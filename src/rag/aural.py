# Standard libs
from pathlib import Path

# Internal libs
from rag.base import BaseRAG
from rag.generation.llm.gemini import GeminiLLM
from rag.retrieval.models.embedder.audio_embedder import AudioEmbedder
from schema import Audio, Context, Query, Response, SearchResult
from utils.ragtools import make_audio_metadatas, make_ids
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

    def index(self, audio_file: Path) -> None:
        embeddings = self.embedder.embed([str(audio_file)])
        ids = make_ids(embeddings)
        metadatas = make_audio_metadatas(embeddings, audio_file)
        self.indexer.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def search(self, query: Query, n_results: int = 5) -> list[SearchResult]:
        if not query.audios:
            return []
        paths = [str(audio.path) for audio in query.audios]
        query_embeddings = self.embedder.embed(paths)
        return self.indexer.search(query_embeddings, n_results)

    def generate(self, query: Query, n_results: int = 3) -> Response:
        results = self.search(query, n_results)
        audios = [Audio(Path(result.payload["audio_path"])) for result in results]
        context = Context(query=query, audios=audios)
        return self.llm.generate(context)
