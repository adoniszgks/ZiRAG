# Standard libs
from pathlib import Path

# 3rdparty libs
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from PIL.Image import Image
from transformers.utils.import_utils import is_flash_attn_2_available as fta2_available

# Internal libs
from config import CACHE_DIR
from rag.retrieval.base import BaseImageRetriever


class ColQwen2Retriever(BaseImageRetriever):
    def __init__(
        self, 
        model_name: str = "vidore/colqwen2-v1.0", 
        cache_dir: Path | None = CACHE_DIR, 
    ) -> None:
        self.model: ColQwen2 = ColQwen2.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2" if fta2_available() else None,
            cache_dir=cache_dir,
            device_map="cuda:0",
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        ).eval()
        
        self.processor: ColQwen2Processor = ColQwen2Processor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )

    def embed_images(self, images: list[Image]) -> torch.Tensor:
        batch = self.processor.process_images(images).to(self.model.device)
        with torch.no_grad():
            return self.model(**batch)
        
    def embed_queries(self, queries: list[str]) -> torch.Tensor:
        batch = self.processor.process_queries(queries).to(self.model.device)
        with torch.no_grad():
            return self.model(**batch)

    def score(self, queries: list[str], images: list[Image]) -> torch.Tensor:
        qs: torch.Tensor = self.embed_queries(queries)
        ps: torch.Tensor = self.embed_images(images)
        return self.processor.score_multi_vector(qs=qs, ps=ps)
