# Standard libs
from dataclasses import dataclass
from typing import Any

Embeddings = list[list[float]]
Metadata = dict[str, Any]


@dataclass
class SearchResult:
    key: str
    score: float
    payload: Metadata


@dataclass
class LLMResponse:
    text: str | None
