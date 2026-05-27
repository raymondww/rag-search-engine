import argparse
import json
import os
import math
from utils import get_stopwords, get_movies
from keyword_search import (
    InvertedIndex, build_index_command, 
    search_command, tokenize_single_term,
    bm25_idf_command, bm25_tf_command)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

stopwords_list = get_stopwords()

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    
    search_vanilla_parser = subparsers.add_parser("search-vanilla", help="Search movies using keyword matching without BM25")
    search_vanilla_parser.add_argument("query", type=str, help="Search query")
    
    build_parser = subparsers.add_parser("build", help="Build and cache the inverted index")
    # TF
    tf_parser = subparsers.add_parser("tf", help="Build and cache the inverted index")
    tf_parser.add_argument("id", type=int, help="Doc ID")
    tf_parser.add_argument("term", type=str, help="Term to calculate TF for")
    # IDF
    idf_parser = subparsers.add_parser("idf", help="calculate IDF for a term")
    idf_parser.add_argument("term", type=str, help="Term to calculate IDF for")
    # TF-IDF
    tf_idf_parser = subparsers.add_parser("tfidf", help="calculate IDF for a term")
    tf_idf_parser.add_argument("id", type=int, help="Doc ID")
    tf_idf_parser.add_argument("term", type=str, help="Term to calculate TF-IDF for")
    # BM25 IDF
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="calculate BM25IDF for a term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to calculate BM25IDF for")
    # BM25 TF
    bm25_tf_parser = subparsers.add_parser(
    "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=InvertedIndex.BM25_K1, help="Tunable BM25 K1 parameter")
    args = parser.parse_args()

    match args.command:
        
        case "search-vanilla":
            data = get_movies()
            result_list = search_command(args.query,data,match_type='exact')
            for i, result in enumerate(result_list, 1):
                print(f"{i}. {result['title']} {result['id']}")
                
        case "search":
            print(f"Loading Index...")
            invertedindex = InvertedIndex()
            try:
                index_dict, docmap_dict,_ = invertedindex.load()
                print("Index loaded successfully.")
            except FileNotFoundError:
                print("Error: index not found. Run `build` first.")
                return
            except EOFError:
                print("Error: index files are empty/corrupted. Re-run `build`.")
                return      
            print(f"Searching for: {args.query}")
            result_list = search_command(args.query,[],match_type='index',index_dict=index_dict,docmap_dict=docmap_dict)
            for i, result in enumerate(result_list, 1):
                print(f"{i}. {result['title']} (id={result['id']})")
        
        case "build":
            print("Building inverted index...")
            build_index_command()
            print("Inverted index built and cached successfully.")

        case "tf":
            invertedindex = InvertedIndex()
            try:
                invertedindex.load()
            except FileNotFoundError:
                print("Error: index not found. Run `build` first.")
                return
            except EOFError:
                print("Error: index files are empty/corrupted. Re-run `build`.")
                return
            term = tokenize_single_term(args.term)
            tf_value = invertedindex.get_tf(args.id, term)
            print(tf_value)
            
        case "idf":
            invertedindex = InvertedIndex()
            try:
                index_dict, docmap_dict, _ = invertedindex.load() 
            except FileNotFoundError:
                print("Error: index not found. Run `build` first.")
                return
            except EOFError:
                print("Error: index files are empty/corrupted. Re-run `build`.")
                return
            term = tokenize_single_term(args.term)
            # document frequency of the term
            term_match_doc_count = len(index_dict.get(term, set()))
            # total number of documents in the corpus
            total_doc_count = len(docmap_dict)
            idf_value = math.log((total_doc_count) / (term_match_doc_count + 1))
            print(f"Inverse document frequency of '{args.term}': {idf_value:.2f}") 
        
        case "tfidf":
            invertedindex = InvertedIndex()
            try:                
                index_dict, docmap_dict, _ = invertedindex.load()
            except FileNotFoundError:
                print("Error: index not found. Run `build` first.")
                return
            except EOFError:
                print("Error: index files are empty/corrupted. Re-run `build`.")
                return
            term = tokenize_single_term(args.term)
            # term frequency of the term in the specified document
            tf_value = invertedindex.get_tf(args.id, term)
            # document frequency of the term
            term_match_doc_count = len(index_dict.get(term, set()))
            # total number of documents in the corpus
            total_doc_count = len(docmap_dict)
            idf_value = math.log((total_doc_count) / (term_match_doc_count + 1))
            tf_idf_value = tf_value * idf_value
            print(f"TF-IDF score of '{args.term}' in document '{args.id}': {tf_idf_value:.2f}")
        
        case "bm25idf":
            try:
                bm25idf = bm25_idf_command(args.term)
            except FileNotFoundError:
                print("Error: index not found. Run `build` first.")
                return
            except EOFError:
                print("Error: index files are empty/corrupted. Re-run `build`.")
                return

            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        
        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

        case _:
            parser.print_help()
            
                      
if __name__ == "__main__":
    main()
