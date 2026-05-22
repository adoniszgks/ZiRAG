# Standard libs
from pathlib import Path

# 3rdparty libs
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rank_bm25 import BM25Okapi

# Internal libs
from config import CACHE_DIR, TEXT_EMB_MODEL
from schema import Metadata, SearchMode, SearchResult
from vectorstore.base import BaseIndexer


class ChromaDBIndexer(BaseIndexer):
    def __init__(self, persist_dir: Path = CACHE_DIR) -> None:
        self.client = PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(
            name="zirag_text",
            embedding_function=SentenceTransformerEmbeddingFunction(TEXT_EMB_MODEL),
        )
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        result = self.collection.get(include=["documents", "metadatas"])
        self._bm25_ids = result["ids"]
        self._bm25_metadatas = result["metadatas"] or []
        docs = result["documents"] or []
        self._bm25 = BM25Okapi([doc.split() for doc in docs]) if docs else None

    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[Metadata] | None = None,
    ) -> None:
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
        self._rebuild_bm25()

    def search(
        self,
        query_text: str,
        n_results: int = 10,
        mode: SearchMode = SearchMode.BM25,
    ) -> list[SearchResult]:
        if mode == SearchMode.BM25:
            return self._bm25_search(query_text, n_results)
        return self._cosine_search(query_text, n_results)

    def _cosine_search(self, query_text: str, n_results: int) -> list[SearchResult]:
        result = self.collection.query(query_texts=[query_text], n_results=n_results)
        ids = (result["ids"] or [[]])[0]
        scores = (result["distances"] or [[]])[0]
        metadatas = (result["metadatas"] or [[]])[0]
        return [
            SearchResult(key=k, score=s, payload=m or {})
            for k, s, m in zip(ids, scores, metadatas)
        ]

    def _bm25_search(self, query_text: str, n_results: int) -> list[SearchResult]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(query_text.split())
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
        return [
            SearchResult(
                key=self._bm25_ids[i],
                score=float(scores[i]),
                payload=self._bm25_metadatas[i] or {},
            )
            for i in top_n
        ]
