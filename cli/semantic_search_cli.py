#!/usr/bin/env python3

import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="verift semantic search model is loaded")
    embed_parser = subparsers.add_parser("embed_text", help="generate embedding for a given text")
    embed_parser.add_argument("text", type=str, help="Text to generate embedding for")
    
    verify_embeddings = subparsers.add_parser("verify_embeddings", help="verify that embeddings can be generated and loaded correctly")
    embed_query = subparsers.add_parser("embed_query", help="generate embedding for a given query")
    embed_query.add_argument("query", type=str, help="Query to generate embedding for")
    search_parser = subparsers.add_parser("search", help="search for documents similar to a given query")
    search_parser.add_argument("query", type=str, help="Query to search for similar documents")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    
    chunk_parser = subparsers.add_parser("chunk", help="chunk a given text into smaller pieces")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Size of each chunk")
    chunk_parser.add_argument("--overlap", type=int, help="Number of overlapping tokens between chunks")

    semantic_chunck_parser = subparsers.add_parser("semantic_chunk", help="perform semantic chuncking on a given query")
    semantic_chunck_parser.add_argument("query", type=str, help="Query to perform semantic chunking on")
    semantic_chunck_parser.add_argument("--max-chunk-size", type=int, default=4, help="Size of each chunk")
    semantic_chunck_parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping tokens between chunks")
    
    embed_chunck_parser = subparsers.add_parser("embed_chunks", help="generate embeddings for chunks of a given query")
    
    search_chuncked_parser = subparsers.add_parser("search_chunked", help="search for documents similar to a given query using chunked embeddings")
    search_chuncked_parser.add_argument("query", type=str, help="Query to search for similar documents")
    search_chuncked_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    
    args = parser.parse_args()

    match args.command:
        case "verify":
            from rag_engine.semantic_search import verify_model
            verify_model()
            
        case "embed_text":
            from rag_engine.semantic_search import embed_text
            embed_text(args.text)
            
        case "verify_embeddings":
            from rag_engine.semantic_search import verify_embeddings
            verify_embeddings()
            
        case "embed_query":
            from utils.search_utils import generate_text
            generate_text(args.query)
            
        case "search":
            from utils.search_utils import get_movies
            from rag_engine.semantic_search import SemanticSearch
 
            movies = get_movies()
            semantic_search = SemanticSearch()
            semantic_search.load_or_create_embeddings(movies)
            results = semantic_search.search(args.query, args.limit)
 
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['description'][:100]}...")
                print()
                
        case "chunk":
            print(f"Chunking {len(args.text)} characters")
            text = args.text.split(" ")
            chunk_size = args.chunk_size
            over_lap = args.overlap if args.overlap is not None else 0
            if over_lap == 0:
                result = [text[t:t+chunk_size] for t in range(0, len(text), chunk_size)]
            if over_lap > 0:
                result = []
                start = 0
                while start < len(text):
                    end = min(start + chunk_size, len(text))
                    result.append(text[start:end])
                    start += chunk_size - over_lap
                    
            for i, chunk in enumerate(result, 1):
                print(f"{i}. {' '.join(chunk)}")
                
        case "semantic_chunk":
            from rag_engine.semantic_search import SemanticSearch
            print(f"Semantic chunking {len(args.query)} characters")
            result = SemanticSearch.semantic_chunking(args.query, args.max_chunk_size, args.overlap)
            
            for i, chunk in enumerate(result, 1):
                print(f"{i}. {' '.join(chunk)}")
        
        case "embed_chunks":
            from rag_engine.semantic_search import ChunkedSemanticSearch
            from utils.search_utils import get_movies
            movies = get_movies()
            chunked_search = ChunkedSemanticSearch()
            embeddings = chunked_search.load_or_create_chunk_embeddings(movies)
            print(f"Generated {len(embeddings)} chunked embeddings")
        
        case "search_chunked":
            from rag_engine.semantic_search import ChunkedSemanticSearch
            from utils.search_utils import get_movies
            movies = get_movies()
            chunked_search = ChunkedSemanticSearch()
            chunked_search.load_or_create_chunk_embeddings(movies)
            results = chunked_search.search_chunks(args.query, args.limit)
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['document']}...")
            
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()