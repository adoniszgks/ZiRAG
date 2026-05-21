# Standard libs
from abc import ABC, abstractmethod
from typing import Any

# 3rdparty libs
from PIL.Image import Image
from torch import Tensor

# Internal libs
from schema import Query


class BaseRetriever(ABC):
    @abstractmethod
    def embed_images(self, images: list[Image]) -> Tensor: ...

    @abstractmethod
    def embed_text(self, text: str) -> Tensor: ...

    @abstractmethod
    def embed_query(self, query: Query) -> Tensor: ...

    @abstractmethod
    def score(self, query: Query, passages: list[Image]) -> Any: ...
