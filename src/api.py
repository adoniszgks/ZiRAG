# 3rdparty libs
from fastapi import FastAPI

# Internal libs
from rag.zirag import ZiRAG


def mount(app: FastAPI, zirag: ZiRAG) -> None:
    @app.get("/api/retrieval-stats")
    def retrieval_stats():
        return zirag.retrieval_log
