# Standard libs
from abc import ABC, abstractmethod
from typing import Any


class BaseIndexer(ABC):
    @abstractmethod
    def add(
        self,
        embeddings: list[list[float]],
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None: ...

    @abstractmethod
    def search(
        self,
        query_embeddings: list[list[float]],
        n_results: int = 10,
    ) -> dict[str, Any]: ...
