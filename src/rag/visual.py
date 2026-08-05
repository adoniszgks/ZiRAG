# Standard libs
from pathlib import Path

# Internal libs
from rag.base import BaseRAG
from rag.retrieval.models.vlm.colqwen2 import ColQwen2Retriever
from schema import Query, SearchResult
from utils.pdftools import convert_pdf_to_pil_images, extract_pdf_texts
from utils.ragtools import make_ids, make_image_metadatas, make_text_metadatas
from vectorstore.base import BaseIndexer


class VisualRAG(BaseRAG):
    def __init__(self, indexer: BaseIndexer, retriever: ColQwen2Retriever) -> None:
        self.indexer = indexer
        self.retriever = retriever

    def index(self, file_path: Path | None) -> None:
        if not file_path:
            return
        texts = extract_pdf_texts(file_path)
        images = convert_pdf_to_pil_images(file_path)
        text_embeddings = self.retriever.embed_texts(texts)
        image_embeddings = self.retriever.embed_images(images)
        embeddings = text_embeddings.tolist() + image_embeddings.tolist()
        ids = make_ids(embeddings)
        metadatas = make_text_metadatas(texts, file_path)
        metadatas += make_image_metadatas(image_embeddings, file_path)
        self.indexer.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def search(self, query: Query, n_results: int) -> list[SearchResult]:
        if not query.images:
            return []
        query_embeddings = self.retriever.embed_images(query.images)[0].tolist()
        return self.indexer.search(query_embeddings, n_results)
