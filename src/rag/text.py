# Standard libs
from pathlib import Path
from uuid import uuid4

# Internal libs
from rag.base import BaseRAG
from schema import Metadata, Query, Response, SearchMode, SearchResult
from utils import pdf2txt
from vectorstore.indexer.chromadb import ChromaDBIndexer


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


class TextRAG(BaseRAG):
    def __init__(self, indexer: ChromaDBIndexer) -> None:
        self.indexer = indexer

    def index(self, pdf_file: Path) -> None:
        texts = pdf2txt.extract_pdf_texts(pdf_file)
        ids = _make_ids(texts)
        metadatas = _make_metadatas(texts, pdf_file)

        self.indexer.add(ids=ids, documents=texts, metadatas=metadatas)

    def generate(self, query: Query, n_results: int = 3) -> Response:
        raise NotImplementedError("TextRAG does not generate — use ZiRAG")

    def search(
        self,
        query: Query,
        n_results: int = 10,
        mode: SearchMode = SearchMode.BM25,
    ) -> list[SearchResult]:
        return self.indexer.search(
            query_text=query.text or "",
            n_results=n_results,
            mode=mode,
        )
