# Standard libs
from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    @abstractmethod
    def score(self, query, documents): ...


class BaseImageRetriever(BaseRetriever):
    @abstractmethod
    def embed_images(self, images): ...
    
    @abstractmethod
    def embed_queries(self, query): ...


class BaseTextRetriever(BaseRetriever):
    @abstractmethod
    def search(self, query: str, corpus: list[str]): ...