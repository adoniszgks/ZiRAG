# Standard libs
from pathlib import Path
from uuid import uuid4

# 3rdparty libs

# Internal libs
from llm.gemini import GeminiClient
from rag.retrieval.base import BaseImageRetriever
from schema import LLMResponse, Metadata, SearchResult
from utils import pdf2img
from vectorstore.base import BaseIndexer


def _make_ids(count: int) -> list[str]:
    return [str(uuid4()) for _ in range(count)]


def _make_metadatas(count: int, pdf_file: Path) -> list[Metadata]:
    return [
        {"page": page_id, "pdf": pdf_file.name, "pdf_path": str(pdf_file)}
        for page_id in range(count)
    ]


class ImageRAGPipeline:
    def __init__(
        self,
        indexer: BaseIndexer,
        retriever: BaseImageRetriever,
        llm: GeminiClient,
    ) -> None:
        self.indexer = indexer
        self.retriever = retriever
        self.llm = llm

    def index(self, pdf_file: Path) -> None:
        images = pdf2img.convert_pdf_to_pil_images(pdf_file)
        multi_embeddings = self.retriever.embed_images(images)
        embeddings = multi_embeddings.tolist()

        count = len(embeddings)
        ids = _make_ids(count)
        metadatas = _make_metadatas(count, pdf_file)

        self.indexer.add(embeddings=embeddings, ids=ids, metadatas=metadatas)

    def search(self, query: str, n_results: int = 10) -> list[SearchResult]:
        multi_embeddings = self.retriever.embed_queries([query])
        query_embeddings = multi_embeddings.tolist()
        return self.indexer.search(query_embeddings, n_results)

    def generate(self, query: str, n_results: int = 3) -> LLMResponse:
        results = self.search(query, n_results)
        images = [
            pdf2img.convert_pdf_page_to_pil_image(
                pdf_file=Path(result.payload["pdf_path"]),
                page_num=result.payload["page"],
            )
            for result in results
        ]
        return self.llm.generate(query, images)
