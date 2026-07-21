# Standard libs
import sys
import webbrowser
from pathlib import Path

# 3rdparty libs
from qdrant_client import QdrantClient

# Internal libs
from config import (
    AUDIO_EMB_DIM,
    CACHE_DIR,
    DATA_DIR,
    LLM_API_KEY,
    LLM_MODEL,
    SYSTEM_PROMPT,
)
from rag.aural import AuralRAG
from rag.generation.llm.gemini import GeminiLLM
from rag.multimodal import MultimodalRAG
from rag.retrieval.models.embedder.audio_embedder import AudioEmbedder
from rag.retrieval.models.embedder.text_embedder import TextEmbedder
from rag.retrieval.models.searcher.bm25 import BM25Searcher
from rag.retrieval.models.vlm.colqwen2 import ColQwen2Retriever
from rag.textual import TextualRAG
from rag.visual import VisualRAG
from ui.app import App
from vectorstore.qdrant import QdrantIndexer

PDF = DATA_DIR / "jur-imp_f5_f50_f505_de.pdf"


def build_llm(model: str = LLM_MODEL, system_prompt: str = SYSTEM_PROMPT) -> GeminiLLM:
    return GeminiLLM(
        api_key=LLM_API_KEY,
        model=model,
        system_prompt=system_prompt,
    )


def build_textual_rag(client: QdrantClient, llm: GeminiLLM) -> TextualRAG:
    return TextualRAG(
        indexer=QdrantIndexer(
            client=client,
            collection_name="textual_collection",
            is_multivector=False,
            vector_size=384,
        ),
        embedder=TextEmbedder(),
        bm25=BM25Searcher(),
        llm=llm,
    )


def build_visual_rag(client: QdrantClient, llm: GeminiLLM) -> VisualRAG:
    try:
        retriever = ColQwen2Retriever(local_files_only=True)
    except Exception:
        retriever = ColQwen2Retriever(local_files_only=False)
    return VisualRAG(
        indexer=QdrantIndexer(client=client, collection_name="visual_collection"),
        retriever=retriever,
        llm=llm,
    )


def build_aural_rag(client: QdrantClient, llm: GeminiLLM) -> AuralRAG:
    return AuralRAG(
        indexer=QdrantIndexer(
            client=client,
            collection_name="aural_collection",
            vector_size=AUDIO_EMB_DIM,
            is_multivector=False,
        ),
        embedder=AudioEmbedder(),
        llm=llm,
    )


def build_mrag(client: QdrantClient, llm: GeminiLLM) -> MultimodalRAG:
    return MultimodalRAG(
        textual_rag=build_textual_rag(client, llm),
        visual_rag=build_visual_rag(client, llm),
        aural_rag=build_aural_rag(client, llm),
        llm=llm,
    )


def index_documents(mrag: MultimodalRAG, paths: list[Path]) -> None:
    mrag.clear()
    for path in paths:
        mrag.index(text=path, image=path, audio=None)


class Main:
    def __init__(self, only_ui: bool = False) -> None:
        client = QdrantClient(path=str(CACHE_DIR / "qdrant"))

        try:
            mrag = None
            if not only_ui:
                llm = build_llm()
                mrag = build_mrag(client, llm)
                paths = [Path(PDF)]
                index_documents(mrag, paths)

            app = App(mrag)
            demo = app.build()
            demo.launch(
                server_name="0.0.0.0",
                prevent_thread_lock=True,
                theme=app.theme,
                css=app.css,
            )
            webbrowser.open("http://127.0.0.1:7860/?__theme=light")
            demo.block_thread()
        finally:
            client.close()


if __name__ == "__main__":
    Main(only_ui="--ui-only" in sys.argv)
