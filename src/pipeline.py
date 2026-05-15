# Standard libs
from pathlib import Path
from typing import Any
from uuid import uuid4

# 3rdparty libs
from PIL.Image import Image
from torch import Tensor

# Internal libs
from rag.retrieval.base import BaseImageRetriever
from utils import pdf2img
from vectorstore.base import BaseIndexer


def _generate_ids(num: int) -> list[str]:
    return [str(uuid4()) for _ in range(num)]

def _generate_metadatas(num: int, pdf_dir: Path) -> list[dict[str, Any]]:
    return [{"page": page_id, "pdf": pdf_dir.name} for page_id in range(num)]

class ImageRAGPipeline:
    def __init__(self, retriever: BaseImageRetriever, indexer: BaseIndexer) -> None:
        self.retriever: BaseImageRetriever = retriever
        self.indexer: BaseIndexer = indexer
    
    def index(self, pdf_dir: Path) -> None:
        images: list[Image] = pdf2img.convert_pdf_to_pil_images(pdf_dir)
        embeddings: list[float] = self.retriever.embed_images(images).tolist()
        num: int = len(embeddings)
        ids: list[str] = _generate_ids(num)
        metadatas: list[dict[str, Any]] = _generate_metadatas(num=num, pdf_dir=pdf_dir)
        self.indexer.add(embeddings=embeddings, ids=ids, metadatas=metadatas)
    
    def query(self, query: str):
        query_embeddings: Tensor = self.retriever.embed_queries([query])
        return self.indexer.search(query_embeddings=query_embeddings)