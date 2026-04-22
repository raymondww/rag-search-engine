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
            # print the search query here
            print(f"Searching for: {args.query}")
            data = read_json(MOVIE_DATA)
            result_list = key_word_search(data,args.query)
            for i,result in enumerate(result_list,1):
                print(f"{i}. {result['title']}")
        case "build":
            print(f"Building...")
            data = read_json(MOVIE_DATA)
            invertedindex = InvertedIndex(data)
            invertedindex.build()
            invertedindex.save()
            
            docs = invertedindex.get_documents("merida")
            print(f"First document for token 'merida' = {docs[0]}")
        case _:
            parser.print_help()

def read_json(file_path:Path) -> list:
    with open(file_path, "r", encoding="utf-8") as f:
        movies = json.load(f)
        movie_list = movies['movies']
        return movie_list
    
def key_word_search(items:list,query:str,) -> list:
        query_token = tokenize_text(preprocessing(query))
        clean_token = remove_stopwords(STOP_WORDS,query_token)
        result = words_matching(clean_token,items)
        return result

def remove_stopwords(file_path:Path,tokens:list) -> list:
    with open(file_path) as f:
        stopwords = [line for line in f if line.strip()]
        # get high-value character and use stem() function to reduce the word to its root form
        kept = [stemmer.stem(w) for w in tokens if w not in stopwords]
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
                      
if __name__ == "__main__":
    main()