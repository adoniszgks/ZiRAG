# Standard libs
from pathlib import Path
from typing import Any
from uuid import uuid4

# Internal libs
from rag.retrieval.base import BaseImageRetriever
from utils import pdf2img
from vectorstore.base import BaseIndexer


def _make_ids(num: int) -> list[str]:
    return [str(uuid4()) for _ in range(num)]


def _make_metadatas(num: int, pdf_path: Path) -> list[dict[str, Any]]:
    return [{"page": page_id, "pdf": pdf_path.name} for page_id in range(num)]


class ImageRAGPipeline:
    def __init__(self, retriever: BaseImageRetriever, indexer: BaseIndexer):
        self.retriever = retriever
        self.indexer = indexer

    def index(self, pdf_path: Path):
        images = pdf2img.convert_pdf_to_pil_images(pdf_path)
        embeddings = self.retriever.embed_images(images).tolist()
        num = len(embeddings)
        ids = _make_ids(num)
        metadatas = _make_metadatas(num=num, pdf_path=pdf_path)
        self.indexer.add(embeddings=embeddings, ids=ids, metadatas=metadatas)

    def query(self, query: str):
        query_embeddings = self.retriever.embed_queries([query])
        return self.indexer.search(query_embeddings=query_embeddings.tolist())
