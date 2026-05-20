# Standard libs
from pathlib import Path

# 3rdparty libs
from chromadb import PersistentClient

# Internal libs
from config import CACHE_DIR
from schema import Embeddings, Metadata, SearchResult
from vectorstore.base import BaseIndexer


class ChromaDBIndexer(BaseIndexer):
    def __init__(
        self,
        collection_name: str = "zirag",
        persist_dir: Path = CACHE_DIR,
    ) -> None:
        self.client = PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add(
        self,
        embeddings: Embeddings,
        ids: list[str],
        metadatas: list[Metadata] | None = None,
    ) -> None:
        self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def search(
        self,
        query_embeddings: Embeddings,
        n_results: int = 10,
    ) -> list[SearchResult]:
        result = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
        )
        ids = result["ids"][0]
        scores = result["distances"][0]
        metadatas = result["metadatas"][0]
        return [
            SearchResult(key=key, score=score, payload=meta or {})
            for key, score, meta in zip(ids, scores, metadatas)
        ]
