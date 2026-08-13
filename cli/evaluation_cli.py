import argparse
import json

def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
         "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    with open("data/golden_dataset.json", "r") as f:
        golden_dataset = json.load(f)
        from utils.search_utils import get_movies
        from rag_engine.hybrid_search import HybridSearch
        movies = get_movies()
        hybridsearch = HybridSearch(movies)
        
    print(f"k={limit}\n")
    for test_case in golden_dataset["test_cases"]:
        rrf_result = hybridsearch.rrf_search(test_case["query"], k=60, limit=limit)
        expected_result = test_case["relevant_docs"]
        relevant_retrieved = len([doc for doc in rrf_result if doc["title"] in expected_result])
        precision = relevant_retrieved / len(rrf_result) if rrf_result else 0
        recall = relevant_retrieved / len(expected_result) if expected_result else 0
        
        print(f"- Query: {test_case['query']}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - Retrieved: {', '.join(doc['title'] for doc in rrf_result)}")
        print(f"  - Relevant: {', '.join(expected_result)}\n")
        
        
if __name__ == "__main__":
    main()