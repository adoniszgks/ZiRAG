# Standard libs
import webbrowser

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
from rag.retrieval.models.embedder.audio_embedder import AudioEmbedder
from rag.retrieval.models.embedder.text_embedder import TextEmbedder
from rag.retrieval.models.searcher.bm25 import BM25Searcher
from rag.retrieval.models.vlm.colqwen2 import ColQwen2Retriever
from rag.textual import TextualRAG
from rag.visual import VisualRAG
from rag.zirag import ZiRAG
from ui.app import App
from vectorstore.qdrant import QdrantIndexer

PDF = DATA_DIR / "jur-imp_f5_f50_f505_de.pdf"


def build_llm() -> GeminiLLM:
    return GeminiLLM(
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )


def build_textual_rag(client: QdrantClient, llm: GeminiLLM) -> TextualRAG:
    return TextualRAG(
        indexer=QdrantIndexer(
            client=client,
            collection_name="zirag_textual",
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
        indexer=QdrantIndexer(client=client, collection_name="zirag_visual"),
        retriever=retriever,
        llm=llm,
    )


def build_aural_rag(client: QdrantClient, llm: GeminiLLM) -> AuralRAG:
    return AuralRAG(
        indexer=QdrantIndexer(
            client=client,
            collection_name="zirag_aural",
            vector_size=AUDIO_EMB_DIM,
            is_multivector=False,
        ),
        embedder=AudioEmbedder(),
        llm=llm,
    )


def build_zirag(client: QdrantClient, llm: GeminiLLM) -> ZiRAG:
    return ZiRAG(
        textual_rag=build_textual_rag(client, llm),
        visual_rag=build_visual_rag(client, llm),
        aural_rag=build_aural_rag(client, llm),
        llm=llm,
    )


class Main:
    def __init__(self) -> None:
        llm = build_llm()
        client = QdrantClient(path=str(CACHE_DIR / "qdrant"))

        try:
            ui = App(
                textual_rag=build_textual_rag(client, llm),
                visual_rag=build_visual_rag(client, llm),
                aural_rag=build_aural_rag(client, llm),
                llm=llm,
            )
            app = ui.build()
            app.launch(
                server_name="0.0.0.0",
                prevent_thread_lock=True,
                theme=ui.theme,
                css=ui.css,
            )
            webbrowser.open("http://127.0.0.1:7860/?__theme=dark")
            app.block_thread()
        finally:
            client.close()


if __name__ == "__main__":
    Main()
