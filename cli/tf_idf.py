from utils import tokenize_text, preprocessing
from pathlib import Path
import os 
import pickle

ROOT = Path(__file__).resolve().parent.parent 

class InvertedIndex():
    def __init__(self, movies: list[dict]):
        """
        Initialize the InvertedIndex with a list of movie documents.

        Parameters
        ----------
        movies : list[dict]
            A list of movie objects (each movie is a dictionary), for example:
            {"id": 4651, "title": "Brave", "description": "..."}.

        Attributes
        ----------
        self.movies
            Stores the raw list of movie documents that will be indexed.

        self.index : dict[str, set[int]]
            The inverted index data structure.

            - Keys are tokens/words (strings), e.g. "merida", "princess"
            - Values are sets of document IDs (integers) for documents containing that token

            Example:
                {
                    "merida": {4651},
                    "princess": {123, 4651, 9002},
                    "dragon": {77, 120}
                }

            This lets us quickly answer:
            "Which documents contain the word X?"

        self.docmap : dict[int, dict]
            A mapping from document ID to the full movie document.

            - Keys are document IDs (ints), e.g. 4651
            - Values are the full movie dictionaries

            Example:
                {
                    4651: {"id": 4651, "title": "Brave", "description": "..."},
                    77:   {"id": 77, "title": "How to Train Your Dragon", "description": "..."}
                }

            This is useful because the inverted index returns only IDs; docmap lets us
            retrieve the full movie data for those IDs after a search.
        """
        
        self.movies = movies
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
    
    def __add_document(self,doc_id,text:str) -> None:
        '''Add each token to the index with the document id'''
        text_tokens = tokenize_text(preprocessing(text))
        for text in text_tokens:
            if text not in self.index:
                self.index[text] = set()
            self.index[text].add(doc_id)
            
    def get_documents(self, term: str) -> list[int]:
        '''Get the document ids for a given term'''
        term = term.lower().strip()
        doc_ids = self.index.get(term, set())
        return sorted(doc_ids)

    def build(self) -> None:
        for m in self.movies:
            doc_id = int(m["id"])
            self.docmap[doc_id] = m  

            text = f"{m.get('title', '')} {m.get('description', '')}"
            self.__add_document(doc_id, text)
            
    def save(self) -> None:
        cache_path = ROOT.joinpath('cache')
        cache_path.mkdir(parents=True, exist_ok=True)
        index_path = cache_path.joinpath('index.pkl')
        docmap_path = cache_path.joinpath('docmap.pkl')
        with index_path.open('wb') as f:
            pickle.dump(self.index, f)
        with docmap_path.open('wb') as f:
            pickle.dump(self.docmap, f)