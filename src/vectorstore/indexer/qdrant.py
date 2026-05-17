# Standard libs
from pathlib import Path
from typing import Any

# 3rdparty libs
from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import QueryResponse
from qdrant_client.models import (
    Distance,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    VectorParams,
)

# Internal libs
from config import CACHE_DIR
from vectorstore.base import BaseIndexer


class QdrantIndexer(BaseIndexer):
    def __init__(
        self,
        collection_name: str = "zirag",
        persist_dir: Path = CACHE_DIR,
        vector_size: int = 128,
    ) -> None:
        self.collection_name: str = collection_name
        self.client: QdrantClient = QdrantClient(path=str(persist_dir / "qdrant"))

        collections_names: list[str] = [
            collection.name for collection in self.client.get_collections().collections
        ]

        if collection_name not in collections_names:
            vectors_config: VectorParams = VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(
                    comparator=MultiVectorComparator.MAX_SIM,
                ),
            )
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
            )

    def add(
        self,
        embeddings: list[list[float]],
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        points: list[PointStruct] = [
            PointStruct(id=point_id, vector=vector, payload=payload)
            for point_id, vector, payload in zip(
                ids,
                embeddings,
                metadatas or [{}] * len(ids),
            )
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(
        self,
        query_embeddings: list[list[float]],
        n_results: int = 10,
    ) -> QueryResponse:
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_embeddings[0],
            limit=n_results,
        )
