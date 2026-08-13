import argparse
import time
import logging

logger = logging.getLogger(__name__)

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for the search pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize documents")
    normalize_parser.add_argument("scores", type=float, nargs="*", help="Scores to normalize")
    
    weigthed_search_parser = subparsers.add_parser("weighted-search", help="Perform weighted hybrid search")
    weigthed_search_parser.add_argument("query", type=str, help="Query string")
    weigthed_search_parser.add_argument("--alpha", type=float, default=0.5, help="Weight for BM25 score (between 0 and 1)")
    weigthed_search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    
    rrf_search_parser = subparsers.add_parser("rrf-search", help="Perform RRF hybrid search")
    rrf_search_parser.add_argument("query", type=str, help="Query string")
    rrf_search_parser.add_argument("-k", type=int, default=5, help="RRF parameter k")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell","rewrite","expand"],
        help="Query enhancement method",
        )    
    rrf_search_parser.add_argument(
        "--rerank-method", 
        type=str, 
        choices=["individual","batch","cross_encoder"], 
        help="Reranking method for RRF search"
        )
    rrf_search_parser.add_argument(
        "--evaluate", 
        action="store_true",
        help="Evaluate the search results using LLM"
        )

    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    if args.debug:
        logging.getLogger("rag_engine").setLevel(logging.DEBUG)
        logging.getLogger("__main__").setLevel(logging.DEBUG)
    
    match args.command:
        case "normalize":
            if args.scores:
                from rag_engine.hybrid_search import normalize_scores
                normalized_scores = normalize_scores(args.scores)
                for score in normalized_scores:
                    print(f"* {score:.4f}")
                
        case "weighted-search":
            from utils.search_utils import get_movies
            from rag_engine.hybrid_search import HybridSearch
            if args.query:
                query = args.query
                alpha = args.alpha
                limit = args.limit

                documents = get_movies()
                hybrid_search = HybridSearch(documents)
                hybrid_results = hybrid_search.weighted_search(query, alpha, limit)
                for i,result in enumerate(hybrid_results):
                    print(f"{i}. {result['title']}")
                    print(f"Hybrid Score: {result['combined_score']:.3f}")
                    print(f"BM25: {result['bm25_score']:.3f}, Semantic: {result['semantic_score']:.3f}")
                    print(result['description'])
                    
        case "rrf-search":
            from utils.search_utils import get_movies
            from rag_engine.hybrid_search import HybridSearch
            if args.query:
                query = args.query
                k = args.k
                limit = args.limit
                documents = get_movies()
                rrf_searches = HybridSearch(documents)
                
                logger.debug("Original query: %r", query)
                
                if args.enhance == "spell":
                    from rag_engine.query_enhancement import spell_check_query
                    query = spell_check_query(query)
                    print(f"Enhanced query (spell): '{args.query}' -> '{query}'\n")
                elif args.enhance == "rewrite":
                    from rag_engine.query_enhancement import rewrite_query
                    query = rewrite_query(query)
                    print(f"Enhanced query (rewrite): '{args.query}' -> '{query}'\n")
                elif args.enhance == "expand":
                    from rag_engine.query_enhancement import expand_query
                    query = expand_query(query)
                    print(f"Enhanced query (expand): '{args.query}' -> '{query}'\n")

                if args.enhance:
                    logger.debug("Query after '%s' enhancement: %r", args.enhance, query)
                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")

                search_limit = limit * 5 if args.rerank_method in ("individual", "batch", "cross_encoder") else limit
                rrf_results = rrf_searches.rrf_search(query, k, search_limit)
                
                logger.debug(
                    "RRF results before re-ranking: %s",
                    [r["title"] for r in rrf_results],
                )

                if args.rerank_method == "individual":
                    from rag_engine.query_enhancement import rerank_results
                    print(f"Re-ranking top {limit} results using {args.rerank_method} method...")
                    for i,doc in enumerate(rrf_results):
                        print(f"Re-ranking {i}/{len(rrf_results)}...")

                        score = rerank_results(query, doc, method="individual")
                        doc["rerank_score"] = float(score)
                        time.sleep(3)
                    rrf_results.sort(key=lambda x: x["rerank_score"], reverse=True)
                    
                elif args.rerank_method == "batch":
                    import json
                    from rag_engine.query_enhancement import rerank_results
                    print(f"Re-ranking top {limit} results using {args.rerank_method} method...")
                    doc_list_str = "\n".join(
                        f"ID {r['doc_id']}: {r['title']} - {r['document'][:200]}"
                        for r in rrf_results
                    )
                    ranking = rerank_results(query, doc_list_str, method="batch")
                    try:
                        json_ranking = json.loads(ranking)
                        rrf_results.sort(key=lambda x: json_ranking.index(x["doc_id"]) if x["doc_id"] in json_ranking else float('inf'))
                    except json.JSONDecodeError:
                        print("Error: Failed to parse JSON ranking. Please check the response format.")
                        return
                
                elif args.rerank_method == "cross_encoder":
                    from sentence_transformers import CrossEncoder
                    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
                    pairs = []
                    for doc in rrf_results:
                        pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])
                    scores = cross_encoder.predict(pairs)
                    for doc, score in zip(rrf_results, scores):
                        doc["cross_encoder_score"] = float(score)
                    rrf_results.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
                    
                rrf_results = rrf_results[:limit]

                print(f"Reciprocal Rank Fusion Results for '{query}' (k={k}):\n")
                for i, result in enumerate(rrf_results, start=1):
                    print(f"{i}. {result['title']}")
                    if args.rerank_method == "individual":
                        print(f"   Re-rank Score: {result['rerank_score']:.3f}/10")
                    elif args.rerank_method == "batch":
                        print(f"   Re-rank Rank: {i}")
                    elif args.rerank_method == "cross_encoder":
                        print(f"   Cross Encoder Score: {scores[rrf_results.index(result)]:.3f}")
                    print(f"   RRF Score: {result['rrf_score']:.3f}")
                    print(f"   BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}")
                    print(f"   {result['document']}")
                    
                if args.evaluate:
                    import json
                    from rag_engine.query_enhancement import evaluate_results
                    evaluation = evaluate_results(query, rrf_results)
                    try:
                        evaluation_json = json.loads(evaluation)
                    except json.JSONDecodeError:
                        print("Error: Failed to parse evaluation JSON. Please check the response format.")
                        return
                    for i, (result, score) in enumerate(zip(rrf_results, evaluation_json), start=1):
                        print(f"{i}. {result['title']}: {score}/3")


        case _:
            parser.print_help()


if __name__ == "__main__":
    main()