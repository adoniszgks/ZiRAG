# Standard libs
from pathlib import Path

# 3rdparty libs
from chromadb import PersistentClient
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Documents, Embeddings, IDs, Metadatas

# Internal libs
from config import CACHE_DIR
from vectorstore.base import BaseIndexer


class ChromaDBIndexer(BaseIndexer):
    def __init__(
        self,
        collection_name: str = "zirag",
        persist_dir: Path = CACHE_DIR,
    ) -> None:
        self.client: ClientAPI = PersistentClient(path=str(persist_dir))
        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add(
        self,
        embeddings: Embeddings,
        ids: IDs,
        metadatas: Metadatas,
    ):
        self.collection.add(
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embeddings: Embeddings,
        query_texts: Documents,
        n_results: int,
    ):
        return self.collection.query(
            query_embeddings=query_embeddings,
            query_texts=query_texts,
            n_results=n_results,
        )
