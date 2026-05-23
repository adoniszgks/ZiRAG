# 3rdparty libs
from rank_bm25 import BM25Okapi

# Internal libs
from schema import Metadata, Query, SearchResult


class BM25Searcher:
    def __init__(self) -> None:
        self.corpus = []
        self.ids = []
        self.metadatas = []
        self.bm25 = None

    def index(
        self,
        ids: list[str],
        metadatas: list[Metadata] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        if not documents:
            return
        self.ids.extend(ids)
        self.corpus.extend(documents)
        self.metadatas.extend(metadatas or [{}] * len(ids))
        self.bm25 = BM25Okapi([document.lower().split() for document in self.corpus])

    def search(self, query: Query, n_results: int = 10) -> list[SearchResult]:
        if not self.bm25:
            return []

        scores = self.bm25.get_scores(query.text.lower().split())
        top_n = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:n_results]

        return [
            SearchResult(
                key=self.ids[i],
                score=float(scores[i]),
                payload=self.metadatas[i],
            )
            for i in top_n
        ]
