# Internal libs
from schema import SearchResult


def _ranks(results: list[SearchResult]) -> dict[str, int]:
    return {result.document_id: rank for rank, result in enumerate(results, start=1)}


def _payloads(results: list[SearchResult]) -> dict[str, dict]:
    return {result.document_id: result.payload for result in results}


def rrf(
    lexical_results: list[SearchResult],
    similarity_results: list[SearchResult],
    n_results: int,
    eta: int = 60,
) -> list[SearchResult]:
    lexical_ranks = _ranks(lexical_results)
    similarity_ranks = _ranks(similarity_results)
    payloads = _payloads(lexical_results) | _payloads(similarity_results)

    scores: dict[str, float] = {}
    for document_id in set(lexical_ranks) | set(similarity_ranks):
        rank_lexical = lexical_ranks.get(document_id)
        rank_similarity = similarity_ranks.get(document_id)

        lexical_score = 1.0 / (eta + rank_lexical) if rank_lexical else 0.0
        similarity_score = 1.0 / (eta + rank_similarity) if rank_similarity else 0.0
        scores[document_id] = lexical_score + similarity_score

    top_n = sorted(scores, key=scores.get, reverse=True)
    return [
        SearchResult(
            document_id=n,
            score=scores[n],
            payload=payloads[n],
        )
        for n in top_n[:n_results]
    ]
