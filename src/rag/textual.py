# Standard libs
from pathlib import Path
from uuid import uuid4

# Internal libs
from rag.base import BaseRAG
from rag.generation.llm.gemini import GeminiLLM
from rag.retrieval.models.embedder.sentence_transformer import TextEmbedder
from rag.retrieval.models.searcher.bm25 import BM25Searcher
from schema import Context, Metadata, Query, Response, SearchMode, SearchResult
from utils import pdftools
from vectorstore.qdrant import QdrantIndexer


def _make_ids(texts: list[str]) -> list[str]:
    return [str(uuid4()) for _ in range(len(texts))]


def _make_metadatas(texts: list[str], pdf_file: Path) -> list[Metadata]:
    return [
        {
            "page": page_id,
            "pdf": pdf_file.name,
            "pdf_path": str(pdf_file),
            "source": "text",
            "text": text,
        }
        for page_id, text in enumerate(texts)
    ]


class TextualRAG(BaseRAG):
    def __init__(
        self,
        indexer: QdrantIndexer,
        embedder: TextEmbedder,
        bm25: BM25Searcher,
        llm: GeminiLLM,
    ) -> None:
        self.indexer = indexer
        self.bm25 = bm25
        self.embedder = embedder
        self.llm = llm

    def index(self, pdf_file: Path) -> None:
        texts = pdftools.extract_pdf_texts(pdf_file)
        embeddings = self.embedder.embed(texts)
        ids = _make_ids(texts)
        metadatas = _make_metadatas(texts, pdf_file)
        self.indexer.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
        self.bm25.index(ids=ids, documents=texts, metadatas=metadatas)

    def search(
        self,
        mode: SearchMode,
        query: Query,
        n_results: int = 10,
    ) -> list[SearchResult]:
        if mode == SearchMode.BM25:
            return self.bm25.search(query, n_results)
        query_embeddings = self.embedder.embed([query.text or ""])
        return self.indexer.search(query_embeddings, n_results)

    def generate(self, query: Query, n_results: int = 3) -> Response:
        results = self.search(mode=SearchMode.SIM, query=query, n_results=n_results)
        texts = [result.payload.get("text", "") for result in results]
        context = Context(query=query, texts=texts, images=[])
        return self.llm.generate(context)
