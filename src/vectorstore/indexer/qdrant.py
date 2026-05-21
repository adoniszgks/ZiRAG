# Standard libs
from pathlib import Path

# 3rdparty libs
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    VectorParams,
)

# Internal libs
from config import CACHE_DIR
from schema import Embedding, Metadata, SearchResult
from vectorstore.base import BaseIndexer


class QdrantIndexer(BaseIndexer):
    def __init__(
        self,
        collection_name: str = "zirag",
        persist_dir: Path = CACHE_DIR,
        vector_size: int = 128,
    ) -> None:
        self.collection_name = collection_name
        self.client = QdrantClient(path=str(persist_dir / "qdrant"))

        collection_names = [
            collection.name for collection in self.client.get_collections().collections
        ]

        if collection_name not in collection_names:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM,
                    ),
                ),
            )

    def add(
        self,
        ids: list[str],
        embeddings: list[Embedding],
        metadatas: list[Metadata] | None = None,
    ) -> None:
        points = [
            PointStruct(id=point_id, vector=vector, payload=payload)
            for point_id, vector, payload in zip(
                ids, embeddings, metadatas or [{}] * len(ids)
            )
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(
        self,
        query_embeddings: list[Embedding],
        n_results: int = 10,
    ) -> list[SearchResult]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embeddings[0],
            limit=n_results,
        )
        return [
            SearchResult(
                key=str(point.id),
                score=point.score,
                payload=point.payload or {},
            )
            for point in response.points
        ]
