# 3rdparty libs
from sentence_transformers import SentenceTransformer

# Internal libs
from config import TEXT_EMB_MODEL


class TextEmbedder:
    def __init__(self, model_name: str = TEXT_EMB_MODEL) -> None:
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()
