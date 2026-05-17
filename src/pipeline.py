# Standard libs
from pathlib import Path
from typing import Any
from uuid import uuid4

# 3rdparty libs
from PIL.Image import Image
from torch import Tensor

# Internal libs
from llm.gemini import GeminiClient
from rag.retrieval.base import BaseImageRetriever
from utils import pdf2img
from vectorstore.base import BaseIndexer


def _make_ids(num: int) -> list[str]:
    return [str(uuid4()) for _ in range(num)]


def _make_metadatas(num: int, pdf_file: Path) -> list[dict[str, Any]]:
    return [
        {"page": page_id, "pdf": pdf_file.name, "pdf_path": str(pdf_file)}
        for page_id in range(num)
    ]


class ImageRAGPipeline:
    def __init__(
        self,
        indexer: BaseIndexer,
        retriever: BaseImageRetriever,
        llm: GeminiClient,
    ) -> None:
        self.indexer: BaseIndexer = indexer
        self.retriever: BaseImageRetriever = retriever
        self.llm: GeminiClient = llm

    def index(self, pdf_file: Path) -> None:
        images: list[Image] = pdf2img.convert_pdf_to_pil_images(pdf_file)
        embeddings: list = self.retriever.embed_images(images).tolist()
        num: int = len(embeddings)
        ids: list[str] = _make_ids(num)
        metadatas: list[dict[str, Any]] = _make_metadatas(num=num, pdf_file=pdf_file)
        self.indexer.add(embeddings=embeddings, ids=ids, metadatas=metadatas)

    def query(self, request: str, n_results: int = 10):
        query_embeddings: Tensor = self.retriever.embed_queries([request])
        return self.indexer.search(
            query_embeddings=query_embeddings.tolist(),
            n_results=n_results,
        )

    def generate(self, query: str, n_results: int = 3) -> str | None:
        results = self.query(query, n_results=n_results)
        images: list[Image] = [
            pdf2img.convert_pdf_page_to_pil_image(
                pdf_file=Path(point.payload["pdf_path"]),
                page_num=point.payload["page"],
            )
            for point in results.points
        ]
        return self.llm.generate(query=query, images=images)
