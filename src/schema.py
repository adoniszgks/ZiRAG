# Standard libs
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# 3rdparty libs
from PIL.Image import Image

Embedding = list[float]
Embeddings = list[Embedding]
Metadata = dict[str, Any]


class SearchMode(Enum):
    BM25 = "keyword"
    SIM = "similarity"
    HYBRID = "hybrid"


@dataclass
class SearchResult:
    key: str
    score: float
    payload: Metadata


@dataclass
class Audio:
    path: Path
    embeddings: list[Embedding] | None = None


@dataclass
class Query:
    texts: list[str] | None = None
    images: list[Image] | None = None
    audios: list[Audio] | None = None


@dataclass
class Context:
    query: Query
    texts: list[str] = field(default_factory=list)
    images: list[Image] = field(default_factory=list)
    audios: list[Audio] = field(default_factory=list)
    language: str = "English"


@dataclass
class Response:
    content: str | None
