# Standard libs
import sys
import webbrowser

sys.path.insert(0, "src")

# 3rdparty libs
from qdrant_client import QdrantClient

# Internal libs
from app import create_app
from config import CACHE_DIR, DATA_DIR, LLM_API_KEY, LLM_MODEL
from rag.generation.llm.gemini import GeminiLLM
from rag.retrieval.models.embedder.sentence_transformer import TextEmbedder
from rag.retrieval.models.searcher.bm25 import BM25Searcher
from rag.retrieval.models.vlm.colqwen2 import ColQwen2Retriever
from rag.textual import TextualRAG
from rag.visual import VisualRAG
from rag.zirag import ZiRAG
from vectorstore.qdrant import QdrantIndexer

PDF = DATA_DIR / "test_manual.pdf"


def build_textual_rag(client: QdrantClient, llm: GeminiLLM) -> TextualRAG:
    return TextualRAG(
        indexer=QdrantIndexer(
            client,
            collection_name="zirag_text",
            is_multivector=False,
            vector_size=384,
        ),
        embedder=TextEmbedder(),
        bm25=BM25Searcher(),
        llm=llm,
    )


def build_visual_rag(client: QdrantClient, llm: GeminiLLM) -> VisualRAG:
    return VisualRAG(
        indexer=QdrantIndexer(client, collection_name="zirag_visual"),
        retriever=ColQwen2Retriever(local_files_only=False),
        llm=llm,
    )


def build_zirag(client: QdrantClient, llm: GeminiLLM) -> ZiRAG:
    return ZiRAG(
        textual_rag=build_textual_rag(client, llm),
        visual_rag=build_visual_rag(client, llm),
        llm=llm,
    )


if __name__ == "__main__":
    llm = GeminiLLM(api_key=LLM_API_KEY, model=LLM_MODEL)
    client = QdrantClient(path=str(CACHE_DIR / "qdrant"))

    try:
        textual_rag = build_textual_rag(client, llm)
        textual_rag.index(PDF)

        app = create_app(rag=textual_rag)
        app.launch(prevent_thread_lock=True)
        webbrowser.open("http://127.0.0.1:7860/?__theme=dark")
        app.block_thread()
    finally:
        client.close()
