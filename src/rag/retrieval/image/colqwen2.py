# Standard libs
from pathlib import Path

# 3rdparty libs
from colpali_engine.models import ColQwen2, ColQwen2Processor
from PIL.Image import Image
from torch import Tensor, bfloat16, cat, no_grad
from transformers import BatchEncoding, BatchFeature
from transformers.utils.import_utils import is_flash_attn_2_available as fta2_available

# Internal libs
from config import CACHE_DIR
from rag.retrieval.base import BaseRetriever
from schema import Query


class ColQwen2Retriever(BaseRetriever):
    def __init__(
        self,
        model_name: str = "vidore/colqwen2-v1.0",
        cache_dir: Path | None = CACHE_DIR,
        device: str = "cuda:0",
        local_files_only: bool = True,
        dtype=bfloat16,
    ) -> None:
        self.model = ColQwen2.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2" if fta2_available() else None,
            cache_dir=cache_dir,
            device_map=device,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        ).eval()
        self.processor = ColQwen2Processor.from_pretrained(model_name, cache_dir)

    def _embed(self, batch: BatchEncoding | BatchFeature) -> Tensor:
        with no_grad():
            return self.model(**batch.to(self.model.device))

    def embed_text(self, text: str) -> Tensor:
        return self._embed(self.processor.process_texts([text]))

    def embed_images(self, images: list[Image]) -> Tensor:
        return self._embed(self.processor.process_images(images))

    def embed_query(self, query: Query) -> Tensor:
        text, images = query.text, query.images
        match (text, images):
            case (None, None):
                raise ValueError("Empty query.")
            case (None, _):
                return self.embed_images(images)
            case (_, None):
                return self.embed_text(text)
        return cat([self.embed_text(text), self.embed_images(images)], dim=1)

    def score(self, query: Query, passages: list[Image]) -> Tensor:
        qs = self.embed_query(query)
        ps = self.embed_images(passages)
        return self.processor.score_multi_vector(qs=qs, ps=ps)
