# Standard libs
from pathlib import Path
from typing import Any

# 3rdparty libs
from chromadb import Collection, PersistentClient

# Internal libs
from config import CACHE_DIR
from vectorstore.base import BaseIndexer


class ChromaDBIndexer(BaseIndexer):
    def __init__(
        self,
        collection_name: str = "zirag",
        persist_dir: Path = CACHE_DIR,
    ) -> None:
        self.client: PersistentClient = PersistentClient(path=str(persist_dir))
        self.collection: Collection = self.client.get_or_create_collection(name=collection_name)

    def add(
        self,
        embeddings: list[list[float]],
        ids: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        self.collection.add(embeddings=embeddings, ids=ids, metadatas=metadatas)

    def search(
        self,
        query_embeddings: list[list[float]] | None = None,
        query_texts: list[str] | None = None,
        n_results: int = 10,
    ) -> dict[str, Any]:
        return self.collection.query(
            query_embeddings=query_embeddings,
            query_texts=query_texts,
            n_results=n_results,
        )
