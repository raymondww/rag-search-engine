import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25_results = self._bm25_search(query, 500 * limit)
        semantic_results = self.semantic_search.search_chunks(query, 500 * limit)

        bm25_scores = normalize_scores([doc["score"] for doc in bm25_results])
        semantic_scores = normalize_scores([doc["score"] for doc in semantic_results])

        combined = {}

        for doc, norm_score in zip(bm25_results, bm25_scores):
            combined[doc["id"]] = {
                "title": doc["title"],
                "doc_id": doc["id"],
                "description": doc["document"],
                "bm25_score": norm_score,
                "semantic_score": 0.0,
            }

        for doc, norm_score in zip(semantic_results, semantic_scores):
            if doc["id"] not in combined:
                combined[doc["id"]] = {
                    "title": doc["title"],
                    "doc_id": doc["id"],
                    "description": doc["document"],
                    "bm25_score": 0.0,
                    "semantic_score": norm_score,
                }
            else:
                combined[doc["id"]]["semantic_score"] = norm_score

        results = []
        for entry in combined.values():
            entry["combined_score"] = hybrid_score(entry["bm25_score"], entry["semantic_score"], alpha)
            results.append(entry)

        return sorted(results, key=lambda x: x["combined_score"], reverse=True)[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        bm25_results = self._bm25_search(query, 500 * limit)
        semantic_results = self.semantic_search.search_chunks(query, 500 * limit)

        combined = {}

        for rank, doc in enumerate(bm25_results, start=1):
            combined[doc["id"]] = {
                "title": doc["title"],
                "doc_id": doc["id"],
                "description": doc["document"],
                "bm25_rank": rank,
                "semantic_rank": None,
                "rrf_score": rrf_score(rank, k),
            }

        for rank, doc in enumerate(semantic_results, start=1):
            if doc["id"] not in combined:
                combined[doc["id"]] = {
                    "title": doc["title"],
                    "doc_id": doc["id"],
                    "description": doc["document"],
                    "bm25_rank": None,
                    "semantic_rank": rank,
                    "rrf_score": rrf_score(rank, k),
                }
            else:
                combined[doc["id"]]["semantic_rank"] = rank
                combined[doc["id"]]["rrf_score"] += rrf_score(rank, k)

        results = list(combined.values())
        return sorted(results, key=lambda x: x["rrf_score"], reverse=True)[:limit]   
        
def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)

def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    normalized_scores = []
    for s in scores:
        normalized_scores.append((s - min_score) / (max_score - min_score))
    return normalized_scores