# Standard libs
from abc import ABC, abstractmethod

from schema import Embedding, Embeddings, Metadata, SearchResult


class BaseIndexer(ABC):
    @abstractmethod
    def add(
        self,
        ids: list[str],
        embeddings: list[Embedding] | list[Embeddings],
        metadatas: list[Metadata],
    ) -> None: ...

    @abstractmethod
    def search(
        self,
        query_embeddings: list[Embedding] | list[Embeddings] | str,
        n_results: int = 10,
    ) -> list[SearchResult]: ...
