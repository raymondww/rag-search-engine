import argparse
import json
from pathlib import Path
from nltk.stem import PorterStemmer
from utils import tokenize_text, preprocessing
from tf_idf import InvertedIndex

stemmer = PorterStemmer()
ROOT = Path(__file__).resolve().parent.parent 
MOVIE_DATA = ROOT.joinpath("data","movies.json")
STOP_WORDS = ROOT.joinpath("data","stopwords.txt")

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    
    build_parser = subparsers.add_parser("build", help="Build and cache the inverted index")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Loading...")
            invertedindex = InvertedIndex()
            try:
                index_dict, docmap_dict = invertedindex.load()
                print("Index loaded successfully.")
            except FileNotFoundError:
                print("Error: index not found. Run `build` first.")
                return
            except EOFError:
                print("Error: index files are empty/corrupted. Re-run `build`.")
                return      
            print(f"Searching for: {args.query}")
            result_list = words_matching_index(args.query,index_dict,docmap_dict)
            # data = read_json(MOVIE_DATA)
            # result_list = key_word_search(data,args.query)
            for i, result in enumerate(result_list, 1):
                print(f"{i}. {result['title']} (id={result['id']})")
                
        case "build":
            print(f"Building...")
            data = read_json(MOVIE_DATA)
            invertedindex = InvertedIndex()
            invertedindex.build(data)
            invertedindex.save()
            print("Inverted index built and cached successfully.")
                        
        case _:
            parser.print_help()

def read_json(file_path:Path) -> list:
    with open(file_path, "r", encoding="utf-8") as f:
        movies = json.load(f)
        movie_list = movies['movies']
        return movie_list

def stemming(filtered_words:list) -> list:
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words

def key_word_search(items:list,query:str,) -> list:
        query_token = tokenize_text(preprocessing(query))
        clean_token = remove_stopwords(STOP_WORDS,query_token)
        stemmed_token = stemming(clean_token)
        result = words_matching(stemmed_token,items)
        return result

def remove_stopwords(file_path:Path,tokens:list) -> list:
    with open(file_path) as f:
        stopwords = {line.strip().lower() for line in f if line.strip()}
        kept = [w for w in tokens if w.lower() not in stopwords]
        return kept
 
def words_matching(query_token:list,items:list):
    result = []
    for item in items:
        title = item.get('title','')
        text_token = tokenize_text(preprocessing(title))
        clean_token = remove_stopwords(STOP_WORDS,text_token)
        for q in query_token:
            for t in clean_token:
                if q in t:
                    result.append(item)  
    return result

def words_matching_index(query: str, index_dict: dict, docmap_dict: dict):
    results = []
    seen = set()

    query_tokens = tokenize_text(preprocessing(query))
    query_tokens = [stemmer.stem(t) for t in query_tokens]

    for token in query_tokens:
        doc_ids = sorted(index_dict.get(token, set()))
        for doc_id in doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)

            doc = docmap_dict.get(doc_id)
            if doc is None:
                continue

            results.append(doc)

            if len(results) >= 5:
                return results

    return results

        
                      
if __name__ == "__main__":
    main()