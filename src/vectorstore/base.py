# Standard libs
from abc import ABC, abstractmethod

from schema import Embeddings, Metadata, SearchResult


class BaseIndexer(ABC):
    @abstractmethod
    def add(
        self,
        embeddings: list[Embeddings],
        ids: list[str],
        metadatas: list[Metadata] | None = None,
    ) -> None: ...

    @abstractmethod
    def search(
        self,
        query_embeddings: list[Embeddings],
        n_results: int = 10,
    ) -> list[SearchResult]: ...
