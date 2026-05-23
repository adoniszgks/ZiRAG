# Standard libs
from pathlib import Path
from uuid import uuid4

# 3rdparty libs
from PIL.Image import Image

# Internal libs
from rag.base import BaseRAG
from rag.generation.llm.gemini import GeminiLLM
from rag.retrieval.models.vlm.colqwen2 import ColQwen2Retriever
from schema import Context, Embedding, Metadata, Query, Response, SearchResult
from utils import pdftools
from vectorstore.base import BaseIndexer


def _make_ids(embeddings: list[Embedding]) -> list[str]:
    return [str(uuid4()) for _ in range(len(embeddings))]


def _make_metadatas(embeddings: list[Embedding], pdf_file: Path) -> list[Metadata]:
    return [
        {
            "page": page_id,
            "pdf": pdf_file.name,
            "pdf_path": str(pdf_file),
            "source": "image",
        }
        for page_id in range(len(embeddings))
    ]


def _make_images(results: list[SearchResult]) -> list[Image]:
    return [
        pdftools.convert_pdf_page_to_pil_image(
            pdf_file=Path(result.payload["pdf_path"]),
            page_num=result.payload["page"],
        )
        for result in results
    ]


class VisualRAG(BaseRAG):
    def __init__(
        self,
        indexer: BaseIndexer,
        retriever: ColQwen2Retriever,
        llm: GeminiLLM,
    ) -> None:
        self.indexer = indexer
        self.retriever = retriever
        self.llm = llm

    def index(self, pdf_file: Path) -> None:
        images = pdftools.convert_pdf_to_pil_images(pdf_file)
        multi_embeddings = self.retriever.embed_images(images)
        embeddings = multi_embeddings.tolist()

        ids = _make_ids(embeddings)
        metadatas = _make_metadatas(embeddings, pdf_file)

        self.indexer.add(embeddings=embeddings, ids=ids, metadatas=metadatas)

    def search(self, query: Query, n_results: int = 10) -> list[SearchResult]:
        query_embeddings = self.retriever.embed_query(query).tolist()
        return self.indexer.search(query_embeddings, n_results)

    def generate(self, query: Query, n_results: int = 3) -> Response:
        results = self.search(query, n_results)
        context = Context(query=query, images=_make_images(results))
        return self.llm.generate(context)
