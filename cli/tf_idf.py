from utils import tokenize_text, preprocessing
from pathlib import Path
import os 
import pickle
from collections import defaultdict
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")

    def build(self,movies:list) -> None:
        for m in movies:
            doc_id = m["id"]
            doc_description = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, doc_description)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
            print(f"Index saved to {self.index_path}")
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
            print(f"Docmap saved to {self.docmap_path}")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(preprocessing(text))
        stemmed = [stemmer.stem(t) for t in tokens]
        for token in set(stemmed):
            self.index[token].add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        term = stemmer.stem(preprocessing(term).strip())
        doc_ids = self.index.get(term, set())
        return sorted(doc_ids)
   
    def load(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.docmap_path):
            raise FileNotFoundError("Index not found. Run the build command first.")

        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)

        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)

        return self.index, self.docmap

        