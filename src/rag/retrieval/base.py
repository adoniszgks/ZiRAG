# Standard libs
from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    @abstractmethod
    def score(self, query, documents): ...


class BaseImageRetriever(BaseRetriever):
    @abstractmethod
    def embed_images(self, images: list) -> list[list[float]]: ...

    @abstractmethod
    def embed_queries(self, queries: list[str]) -> list[list[float]]: ...


class BaseTextRetriever(BaseRetriever):
    @abstractmethod
    def search(self, query: str, corpus: list[str]): ...
