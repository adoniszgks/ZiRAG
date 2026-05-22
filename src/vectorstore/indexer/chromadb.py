# Standard libs
from pathlib import Path

# 3rdparty libs
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import (
    ChromaBm25EmbeddingFunction,
    SentenceTransformerEmbeddingFunction,
)

# Internal libs
from config import CACHE_DIR, TEXT_EMB_MODEL
from schema import Metadata, SearchMode, SearchResult
from vectorstore.base import BaseIndexer


class ChromaDBIndexer(BaseIndexer):
    def __init__(self, persist_dir: Path = CACHE_DIR) -> None:
        self.client = PersistentClient(path=str(persist_dir))
        self.cosine_collection = self.client.get_or_create_collection(
            name="zirag_cosine",
            embedding_function=SentenceTransformerEmbeddingFunction(TEXT_EMB_MODEL),
        )
        self.bm25_collection = self.client.get_or_create_collection(
            name="zirag_bm25",
            embedding_function=ChromaBm25EmbeddingFunction(),
        )

    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[Metadata] | None = None,
    ) -> None:
        self.cosine_collection.add(ids=ids, documents=documents, metadatas=metadatas)
        self.bm25_collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def search(
        self,
        query_text: str,
        n_results: int = 10,
        mode: SearchMode = SearchMode.BM25,
    ) -> list[SearchResult]:
        collection = (
            self.bm25_collection if mode == SearchMode.BM25 else self.cosine_collection
        )
        result = collection.query(query_texts=[query_text], n_results=n_results)
        ids = (result["ids"] or [[]])[0]
        scores = (result["distances"] or [[]])[0]
        metadatas = (result["metadatas"] or [[]])[0]
        return [
            SearchResult(key=key, score=score, payload=payload or {})
            for key, score, payload in zip(ids, scores, metadatas)
        ]
