from utils import tokenize_text, preprocessing, stemming
from pathlib import Path
import os 
import math
import pickle
from collections import defaultdict,Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_freq = defaultdict(Counter)
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_freq_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")

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
        with open(self.term_freq_path, "wb") as f:
            pickle.dump(self.term_freq, f)
            print(f"Term Freq saved to {self.term_freq_path}")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(preprocessing(text))
        # stemmed = [stemmer.stem(t) for t in tokens]
        stemmed = stemming(tokens)

        counts = Counter(stemmed)
        self.term_freq[doc_id] = counts

        for token in counts.keys():
            self.index[token].add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        tokens = tokenize_text(preprocessing(term))
        token = stemming(tokens)[0]
        # term = stemmer.stem(preprocessing(term).strip())
        doc_ids = self.index.get(token, set())
        return sorted(doc_ids)
   
    def load(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.docmap_path):
            raise FileNotFoundError("Index not found. Run the build command first.")

        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)

        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
            
        with open(self.term_freq_path, "rb") as f:
                self.term_freq = pickle.load(f)

        return self.index, self.docmap, self.term_freq

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize_text(preprocessing(term))
        if len(tokens) != 1:
            raise ValueError(f"Expected exactly 1 token, got {len(tokens)}: {tokens}")
        token = stemming(tokens)[0]
        return self.term_freq.get(doc_id, Counter()).get(token, 0)
    
    def get_bm25_idf(self, term:str) -> float:
        # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
        tokens = tokenize_text(preprocessing(term))
        if len(tokens) != 1:
            raise ValueError(f"Expected exactly 1 token, got {len(tokens)}: {tokens}")
        token = stemming(tokens)[0]
        df = len(self.index.get(token, set()))
        N = len(self.docmap)
        return math.log((N - df + 0.5) / (df + 0.5) + 1)
