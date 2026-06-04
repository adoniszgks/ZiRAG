# Standard libs
from pathlib import Path

# Internal libs
from rag.base import BaseRAG
from rag.generation.llm.gemini import GeminiLLM
from rag.retrieval.models.embedder.text_embedder import TextEmbedder
from rag.retrieval.models.searcher.bm25 import BM25Searcher
from schema import Context, Query, Response, SearchMode, SearchResult
from utils import pdftools
from utils.ragtools import make_ids, make_text_metadatas
from vectorstore.qdrant import BaseIndexer


class TextualRAG(BaseRAG):
    def __init__(
        self,
        indexer: BaseIndexer,
        embedder: TextEmbedder,
        bm25: BM25Searcher,
        llm: GeminiLLM,
    ) -> None:
        self.indexer = indexer
        self.bm25 = bm25
        self.embedder = embedder
        self.llm = llm

    def index(self, pdf_file: Path | None) -> None:
        if not pdf_file:
            return
        texts = pdftools.extract_pdf_texts(pdf_file)
        embeddings = self.embedder.embed(texts)
        ids = make_ids(texts)
        metadatas = make_text_metadatas(texts, pdf_file)
        self.indexer.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
        self.bm25.index(ids=ids, documents=texts, metadatas=metadatas)

    def search(
        self,
        mode: SearchMode,
        query: Query,
        n_results: int = 10,
    ) -> list[SearchResult]:
        if not query.texts:
            return []
        if mode == SearchMode.BM25:
            return self.bm25.search(query, n_results)
        query_embeddings = self.embedder.embed(query.texts)
        return self.indexer.search(query_embeddings, n_results)

    def generate(self, query: Query, n_results: int = 3) -> Response:
        results = self.search(mode=SearchMode.SIM, query=query, n_results=n_results)
        texts = [result.payload.get("text", "") for result in results]
        context = Context(query=query, texts=texts, images=[])
        return self.llm.generate(context)
