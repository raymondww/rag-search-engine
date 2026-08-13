import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize a given text"
    )
    summarize_parser.add_argument("query", type=str, help="Text to summarize")
    summarize_parser.add_argument(
        "--limit", type=int, default=5, help="Number of search results to consider for summarization"
    )
    
    citation_parser = subparsers.add_parser(
        "citations", help="Generate citations for a given query"
    )
    citation_parser.add_argument("query", type=str, help="Query for citation generation")
    citation_parser.add_argument("--limit", type=int, default=5, help="Number of search results to consider for citation generation")
    
    answer_parser = subparsers.add_parser(
        "question", help="Generate an answer for a given query"
    )
    answer_parser.add_argument("query", type=str, help="Query for answer generation")
    answer_parser.add_argument("--limit", type=int, default=5, help="Number of search results to consider for answer generation")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            # do RAG stuff here
            from rag_engine.query_enhancement import call_rag_agent
            from utils.search_utils import get_movies
            from rag_engine.hybrid_search import HybridSearch
            movies = get_movies()
            hybrid_search = HybridSearch(movies)
            rrf_results = hybrid_search.rrf_search(query, k=60, limit=5)
            
            doc_list_str = "\n".join(
                f"[{i}] {r['title']} - {r['document'][:200]}"
                for i, r in enumerate(rrf_results, start=1)
            )
            results = call_rag_agent(query, doc_list_str)
            print("Search Results:\n")
            for result in rrf_results:
                print(f"- {result['title']}")
            print("RAG Response:\n")
            print(results)
        
        case "summarize":
            from utils.search_utils import get_movies
            from rag_engine.hybrid_search import HybridSearch
            from rag_engine.query_enhancement import summarize_text
            query = args.query
            limit = args.limit
            movies = get_movies()
            hybrid_search = HybridSearch(movies)
            rrf_results = hybrid_search.rrf_search(query, k=60, limit=limit)
            doc_list_str = "\n".join(
                f"[{i}] {r['title']} - {r['document'][:200]}"
                for i, r in enumerate(rrf_results, start=1)
            )
            results = summarize_text(query,doc_list_str)
            print("Search Results:")
            for result in rrf_results:
                print(f"- {result['title']}")
            print("\nLLM Summary:")
            print(results)
            
        case "citations":
            from utils.search_utils import get_movies
            from rag_engine.hybrid_search import HybridSearch
            from rag_engine.query_enhancement import generate_citations
            query = args.query
            limit = args.limit
            movies = get_movies()
            hybrid_search = HybridSearch(movies)
            rrf_results = hybrid_search.rrf_search(query, k=60, limit=limit)
            doc_list_str = "\n".join(
                f"[{i}] {r['title']} - {r['document'][:200]}"
                for i, r in enumerate(rrf_results, start=1)
            )
            results = generate_citations(query,doc_list_str)
            print("Search Results:\n")
            for result in rrf_results:
                print(f"  - {result['title']}")
            print("\nLLM Answer:")
            print(results)
            
        case "question":
            from utils.search_utils import get_movies
            from rag_engine.hybrid_search import HybridSearch
            from rag_engine.query_enhancement import call_rag_agent
            query = args.query
            limit = args.limit
            movies = get_movies()
            hybrid_search = HybridSearch(movies)
            rrf_results = hybrid_search.rrf_search(query, k=60, limit=limit)
            doc_list_str = "\n".join(
                f"[{i}] {r['title']} - {r['document'][:200]}"
                for i, r in enumerate(rrf_results, start=1)
            )
            results = call_rag_agent(query,doc_list_str)
            print("Search Results:\n")
            for result in rrf_results:
                print(f"  - {result['title']}")
            print("\nAnswer:")
            print(results)
            
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()