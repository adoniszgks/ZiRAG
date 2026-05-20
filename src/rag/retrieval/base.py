# Standard libs
from abc import ABC, abstractmethod
from typing import Any


class BaseRetriever(ABC):
    @abstractmethod
    def score(self, query, documents) -> Any: ...


class BaseImageRetriever(BaseRetriever):
    @abstractmethod
    def embed_images(self, images) -> Any: ...

    @abstractmethod
    def embed_text(self, text) -> Any: ...

    @abstractmethod
    def embed_query(self, query) -> Any: ...


class BaseTextRetriever(BaseRetriever):
    @abstractmethod
    def search(self, query) -> Any: ...
