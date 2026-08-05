# Standard libs
from abc import ABC, abstractmethod

# Internal libs
from schema import Query, SearchResult


class BaseRAG(ABC):
    @abstractmethod
    def index(self, file_path) -> None: ...

    @abstractmethod
    def search(self, query: Query, n_results: int) -> list[SearchResult]: ...

