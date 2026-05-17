# Standard libs
from abc import ABC, abstractmethod
from typing import Any


class BaseIndexer(ABC):
    @abstractmethod
    def add(self, embeddings, ids, metadatas=None) -> None: ...

    @abstractmethod
    def search(self, query_embeddings, n_results=10) -> Any: ...
