# Standard libs
from pathlib import Path

# Internal libs
from rag.base import BaseRAG
from rag.generation.llm.gemini import GeminiLLM
from rag.retrieval.models.vlm.colqwen2 import ColQwen2Retriever
from schema import Context, Query, Response, SearchResult
from utils.pdftools import convert_pdf_to_pil_images
from utils.ragtools import make_ids, make_image_metadatas, make_images
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

    def index(self, image_path: Path | None) -> None:
        if not image_path:
            return
        images = convert_pdf_to_pil_images(image_path)
        multi_embeddings = self.retriever.embed_images(images)
        embeddings = multi_embeddings.tolist()
        ids = make_ids(embeddings)
        metadatas = make_image_metadatas(embeddings, image_path)
        self.indexer.add(embeddings=embeddings, ids=ids, metadatas=metadatas)

    def search(self, query: Query, n_results: int = 10) -> list[SearchResult]:
        if not query.texts and not query.images:
            return []
        query_embeddings = self.retriever.embed_query(query)[0].tolist()
        return self.indexer.search(query_embeddings, n_results)

    def generate(self, query: Query, n_results: int = 3) -> Response:
        results = self.search(query, n_results)
        context = Context(query=query, images=make_images(results))
        return self.llm.generate(context)
