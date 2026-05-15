# Standard libs
from pathlib import Path

# 3rdparty libs 
from chromadb import PersistentClient
from chromadb.api.models.Collection import Collection
from chromadb.types import Embeddings, IDs, Metadatas, QueryResult

# Internal libs
from vectorstore.base import BaseIndexer


class ChromaDBIndexer(BaseIndexer):
    def __init__(self, collection_name: str, persist_dir: Path) -> None:
        self.client: PersistentClient = PersistentClient(path=persist_dir)
        self.collection: Collection = self.client.get_or_create_collection(name=collection_name)
        
    def add(self, embeddings: Embeddings, ids: IDs, metadatas: Metadatas) -> None:
        self.collection.add(
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
        
    def search(
        self,
        query_embeddings: list[Embeddings] | None = None,
        query_texts: list[str] | None = None,
        n_results: int = 10,
    ) -> QueryResult:
        return self.collection.query(
            query_embeddings=query_embeddings,
            query_texts=query_texts,
            n_results=n_results,
        )