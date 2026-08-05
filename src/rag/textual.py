# Standard libs
from pathlib import Path

# Internal libs
from rag.base import BaseRAG
from rag.retrieval.models.embedder.text_embedder import TextEmbedder
from rag.retrieval.models.searcher.bm25 import BM25Searcher
from rag.retrieval.models.searcher.hybrid import rrf
from schema import Query, SearchResult
from utils.pdftools import extract_pdf_texts
from utils.ragtools import make_ids, make_text_metadatas
from vectorstore.base import BaseIndexer


class TextualRAG(BaseRAG):
    def __init__(
        self,
        indexer: BaseIndexer,
        embedder: TextEmbedder,
        bm25: BM25Searcher,
    ) -> None:
        self.indexer = indexer
        self.bm25 = bm25
        self.embedder = embedder

    def index(self, file_path: Path | None) -> None:
        if not file_path:
            return
        texts = extract_pdf_texts(file_path)
        embeddings = self.embedder.embed(texts)
        ids = make_ids(texts)
        metadatas = make_text_metadatas(texts, file_path)
        self.indexer.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
        self.bm25.index(ids=ids, documents=texts, metadatas=metadatas)

    def search(self, query: Query, n_results: int) -> list[SearchResult]:
        if not query.texts:
            return []
        lexical_results = self.bm25.search(query, n_results)
        query_embeddings = self.embedder.embed(query.texts)
        similarity_results = self.indexer.search(query_embeddings, n_results)
        return rrf(lexical_results, similarity_results, n_results)
