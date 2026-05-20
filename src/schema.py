# Standard libs
from dataclasses import dataclass
from typing import Any

# 3rdparty libs
from PIL.Image import Image

Embeddings = list[list[float]]
Metadata = dict[str, Any]


@dataclass
class SearchResult:
    key: str
    score: float
    payload: Metadata


@dataclass
class Query:
    text: str | None = None
    images: list[Image] | None = None


@dataclass
class Context:
    query: Query
    images: list[Image]


@dataclass
class Response:
    text: str | None
