# Standard libs
from pathlib import Path

# 3rdparty libs
from PIL.Image import Image

# Internal libs
from rag.base import BaseRAG
from rag.generation.llm.gemini import GeminiLLM
from rag.retrieval.models.vlm.colqwen2 import ColQwen2Retriever
from schema import Context, Query, Response, SearchResult
from utils import pdftools
from utils.ragtools import make_ids, make_image_metadatas
from vectorstore.base import BaseIndexer


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

    @staticmethod
    def _make_images(results: list[SearchResult]) -> list[Image]:
        return [
            pdftools.convert_pdf_page_to_pil_image(
                pdf_file=Path(result.payload["pdf_path"]),
                page_num=result.payload["page"],
            )
            for result in results
        ]

    def index(self, pdf_file: Path | None) -> None:
        if not pdf_file:
            return
        images = pdftools.convert_pdf_to_pil_images(pdf_file)
        multi_embeddings = self.retriever.embed_images(images)
        embeddings = multi_embeddings.tolist()
        ids = make_ids(embeddings)
        metadatas = make_image_metadatas(embeddings, pdf_file)
        self.indexer.add(embeddings=embeddings, ids=ids, metadatas=metadatas)

    def search(self, query: Query, n_results: int = 10) -> list[SearchResult]:
        if not query.texts and not query.images:
            return []
        query_embeddings = self.retriever.embed_query(query).tolist()
        return self.indexer.search(query_embeddings, n_results)

    def generate(self, query: Query, n_results: int = 3) -> Response:
        results = self.search(query, n_results)
        context = Context(query=query, images=self._make_images(results))
        return self.llm.generate(context)
