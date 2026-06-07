# Standard libs
from abc import ABC, abstractmethod

# Internal libs
from schema import Query, Response, SearchResult


class BaseRAG(ABC):
    @abstractmethod
    def index(self, file_path) -> None: ...

    @abstractmethod
    def search(self, query: Query, n_results: int) -> list[SearchResult]: ...

    @abstractmethod
    def generate(self, query: Query, n_results: int) -> Response: ...
