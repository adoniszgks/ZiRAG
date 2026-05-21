# Internal libs
from app import create_app
from llm.gemini import GeminiClient
from pipeline import ZiRAG
from rag.retrieval.audio.whisper_stt import WhisperTranscriber
from rag.retrieval.image.colqwen2 import ColQwen2Retriever
from vectorstore.indexer.qdrant import QdrantIndexer
from config import LLM_API_KEY, LLM_MODEL

if __name__ == "__main__":
    retriever = ColQwen2Retriever()
    indexer = QdrantIndexer()
    llm = GeminiClient(api_key=LLM_API_KEY, model=LLM_MODEL)
    zirag = ZiRAG(indexer=indexer, retriever=retriever, llm=llm)
    transcriber = WhisperTranscriber()

    app = create_app(zirag=zirag, transcriber=transcriber)
    app.launch(share=True)
