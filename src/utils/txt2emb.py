# 3rdparty libs
from sentence_transformers import SentenceTransformer

# Internal libs
from config import TEXT_EMBEDDING_MODEL
from schema import Embedding


class TextEmbedder:
    def __init__(self, model_name: str = TEXT_EMBEDDING_MODEL) -> None:
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[Embedding]:
        return self.model.encode(texts).tolist()
