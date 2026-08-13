"""
Diagnostic script: run this in your project root with:
    uv run python3 debug_ranking.py

It shows exactly where "Eliminators" falls in the BM25 list, the semantic
(chunk) list, and the final RRF-combined list for the query
"action movie with lasers" - so we can see which stage is dropping it.
"""
from utils.search_utils import get_movies
from rag_engine.hybrid_search import HybridSearch, combine_rrf_results

QUERY = "action movie with lasers"
TARGET_TITLE = "Eliminators"

movies = get_movies()
hs = HybridSearch(movies)

bm25_results = hs._bm25_search(QUERY, 2500)
semantic_results = hs.semantic_search.search_chunks(QUERY, 2500)

def find(results, title):
    for i, r in enumerate(results, start=1):
        if r["title"] == title:
            return i, r["score"]
    return None, None

bm25_rank, bm25_score = find(bm25_results, TARGET_TITLE)
sem_rank, sem_score = find(semantic_results, TARGET_TITLE)

print(f"BM25 rank for {TARGET_TITLE!r}: {bm25_rank} (score={bm25_score})")
print(f"Semantic rank for {TARGET_TITLE!r}: {sem_rank} (score={sem_score})")

print("\nTop 10 BM25:")
for i, r in enumerate(bm25_results[:10], start=1):
    print(f"  {i}. {r['title']} ({r['score']})")

print("\nTop 10 Semantic:")
for i, r in enumerate(semantic_results[:10], start=1):
    print(f"  {i}. {r['title']} ({r['score']})")

combined = combine_rrf_results(bm25_results, semantic_results, k=5, limit=10)
print("\nTop 10 RRF combined (k=5):")
for i, r in enumerate(combined, start=1):
    print(f"  {i}. {r['title']} rrf={r['rrf_score']:.4f} bm25_rank={r['bm25_rank']} sem_rank={r['semantic_rank']}")
