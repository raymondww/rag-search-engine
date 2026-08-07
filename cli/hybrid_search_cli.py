import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
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
    
    args = parser.parse_args()

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
                if args.enhance == "spell":
                    from rag_engine.llm_check import spell_check_query
                    enhanced_query = spell_check_query(query)
                    print(f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced_query}'\n")
                if args.enhance == "rewrite":
                    from rag_engine.llm_check import rewrite_query
                    enhanced_query = rewrite_query(query)
                    print(f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced_query}'\n")
                if args.enhance == "expand":
                    from rag_engine.llm_check import expand_query
                    enhanced_query = expand_query(query)
                    print(f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced_query}'\n")
                    
                documents = get_movies()
                rrf_searches = HybridSearch(documents)
                rrf_results = rrf_searches.rrf_search(query, k, limit)
                for i,result in enumerate(rrf_results):
                    print(f"{i}. {result['title']}")
                    print(f"RRF Score: {result['rrf_score']:.3f}")
                    print(f"BM25: {result['bm25_rank']}, Semantic: {result['semantic_rank']}")
                    print(result['description'])
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()