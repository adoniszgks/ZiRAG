# Standard libs
from abc import ABC, abstractmethod

from schema import Embedding, Embeddings, Metadata, SearchResult


class BaseIndexer(ABC):
    @abstractmethod
    def add(
        self,
        ids: list[str],
        embeddings: list[Embedding] | list[Embeddings] | None = None,
        metadatas: list[Metadata] | None = None,
    ) -> None: ...

    @abstractmethod
    def search(
        self,
        query_embeddings: list[Embedding] | list[Embeddings] | None = None,
        n_results: int = 10,
    ) -> list[SearchResult]: ...
