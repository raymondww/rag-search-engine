import os 
import math
import pickle
from typing import Literal
from collections import defaultdict, Counter
from utils.search_utils import get_stopwords, normalize_tokens, get_movies, preprocessing

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

stopwords_list = get_stopwords()

class InvertedIndex:
    BM25_K1 = 1.5
    BM25_B = 0.75
    def __init__(self) -> None:
        # {'fruits': {'1', '2'}})
        self.index = defaultdict(set)
        # {1: {'title': 'The Apple', 'description': 'A movie about an apple.'}, 2: {...}, ...}
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_freq = defaultdict(Counter)
        self.term_freq_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths = defaultdict(int)
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")
        self._avg_doc_length: float | None = None
        
    def build(self) -> None:
        """
        Build the inverted index from a list of movie documents.

        Parameters
        ----------
        movies : list
            List of movie dicts, each containing at least 'id', 'title',
            and 'description' keys.

        Returns
        -------
        None
            Populates ``self.docmap`` and the inverted index in-place.

        Notes
        -----
        Each document's searchable text is formed by concatenating its title
        and description. The original document is preserved in ``self.docmap``
        keyed by doc ID, while the preprocessed tokens are stored in the
        inverted index via ``__add_document``.
        """
        movies = get_movies()
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
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)
            print(f"Doc Lengths saved to {self.doc_lengths_path}")
            
    def __add_document(self, doc_id: int, text: str) -> None:
        """
        Preprocess and index a single document into the inverted index.

        Parameters
        ----------
        doc_id : int
            Unique identifier for the document, used as the value in the index.
        text : str
            Raw text to index, typically a concatenation of title and description.

        Returns
        -------
        None
            Updates ``self.index`` in-place by adding ``doc_id`` to the set
            of each token extracted from ``text``.

        Notes
        -----
        The preprocessing pipeline applied to ``text`` is:

        1. ``preprocessing``  — lowercase and remove punctuation
        2. ``tokenize_text``  — split into tokens on whitespace
        3. stop word removal  — filter tokens against ``STOP_WORDS``
        4. ``stemming``       — reduce tokens to their root form

        The same pipeline must be applied to search queries to ensure
        consistent matching against this index.
        """
        tokenize_query = normalize_tokens(text, stopwords=stopwords_list, stem=True)
        for token in set(tokenize_query):
            self.index[token].add(doc_id)
        # counts = Counter(tokenize_query)
        self.term_freq[doc_id].update(tokenize_query)
        self.doc_lengths[doc_id] = len(tokenize_query)

    def get_documents(self, term: str) -> list[int]:
        """Get all document IDs containing a term.
        
        Parameters
        ----------
        term : str
            Term to search for (will be normalized using the same pipeline as indexing).
        
        Returns
        -------
        list[int]
            Sorted list of document IDs.
        """
        tokens = normalize_tokens(term, stopwords=stopwords_list, stem=True)
        if not tokens:
            return []
        # If multiple tokens, search for all and return docs containing any
        doc_ids = set()
        for token in tokens:
            doc_ids.update(self.index.get(token, set()))
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
                
        with open(self.doc_lengths_path, "rb") as f:
                self.doc_lengths = pickle.load(f)
    
    def get_tf(self, doc_id: int, term: str) -> int:
        term = tokenize_single_term(term)
        return self.term_freq[doc_id][term]
    
    def get_bm25_idf(self, term:str) -> float:
        # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
        term = tokenize_single_term(term)
        term_match_doc_count = len(self.index.get(term, set()))
        total_doc_count = len(self.docmap)
        return math.log((total_doc_count - term_match_doc_count + 0.5) / (term_match_doc_count + 0.5) + 1)
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B ) -> float:
        # Length normalization factor
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)
    
    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)
        
    def bm25(self, doc_id, term) -> float:
        idf = self.get_bm25_idf(term)
        tf = self.get_bm25_tf(doc_id, term)
        return idf * tf
    
    def bm25_search(self, query, limit):
        tokenize_query = normalize_tokens(query, stopwords=stopwords_list, stem=True)
        doc_scores = defaultdict(float)
        for token in tokenize_query:
            for doc_id in self.index.get(token, set()):
                doc_scores[doc_id] += self.bm25(doc_id, token)
                
        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in ranked_docs[:limit]:
            title = self.docmap.get(doc_id, {}).get("title", "")
            # debug breakdown
            print(f"\n{'='*40}")
            print(f"Doc {doc_id}: {title}")
            print(f"  Total BM25 Score: {score:.4f}")
            print(f"docs containing 'anim': {len(self.index.get('anim', set()))}")
            print(f"docs containing 'famili': {len(self.index.get('famili', set()))}")
            print(f"total docs: {len(self.docmap)}")
            for token in tokenize_query:
                idf = self.get_bm25_idf(token)
                tf = self.get_bm25_tf(doc_id, token)
                raw_tf = self.get_tf(doc_id, token)
                doc_len = self.doc_lengths.get(doc_id, 0)
                avg_len = self.__get_avg_doc_length()
                print(f"  Token: '{token}'")
                print(f"    raw TF:     {raw_tf}")
                print(f"    BM25 TF:    {tf:.4f}")
                print(f"    BM25 IDF:   {idf:.4f}")
                print(f"    BM25 score: {idf * tf:.4f}")
                print(f"    doc_len:    {doc_len}, avg_len: {avg_len:.2f}")
                # if doc_id == 1907:
                #     print(f"doc_length: {self.doc_lengths.get(1907)}")
                #     print(f"term_freq: {self.term_freq.get(1907)}")
            results.append({"id": doc_id, "title": title, "score": score})
        return results
    def search_with_trace(self, query: str, limit: int = 5) -> dict:
        lowered = query.lower()
        stripped = preprocessing(lowered)
        tokens = stripped.split()
        no_stopwords = [t for t in tokens if t not in stopwords_list]
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        stemmed = [stemmer.stem(t) for t in no_stopwords]

        term_lookups = []
        for term in stemmed:
            doc_ids = sorted(self.index.get(term, set()))
            idf = self.get_bm25_idf(term)
            term_lookups.append({
                "term": term,
                "matching_doc_ids": doc_ids,
                "doc_frequency": len(doc_ids),
                "idf": round(idf, 4),
            })

        doc_scores = defaultdict(float)
        doc_breakdowns = defaultdict(list)
        for term in stemmed:
            idf = self.get_bm25_idf(term)
            for doc_id in self.index.get(term, set()):
                raw_tf = self.term_freq[doc_id][term]
                bm25_tf = self.get_bm25_tf(doc_id, term)
                contribution = idf * bm25_tf
                doc_scores[doc_id] += contribution
                doc_breakdowns[doc_id].append({
                    "term": term,
                    "raw_tf": raw_tf,
                    "bm25_tf": round(bm25_tf, 4),
                    "idf": round(idf, 4),
                    "contribution": round(contribution, 4),
                })

        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        results = []
        for doc_id, score in ranked:
            movie = self.docmap.get(doc_id, {})
            results.append({
                "doc_id": doc_id,
                "title": movie.get("title", ""),
                "description": movie.get("description", ""),
                "score": round(score, 4),
                "breakdown": doc_breakdowns[doc_id],
            })

        return {
            "original": query,
            "lowercased": lowered,
            "stripped": stripped,
            "tokens": tokens,
            "no_stopwords": no_stopwords,
            "stemmed": stemmed,
            "term_lookups": term_lookups,
            "results": results,
        }
    
def tokenize_single_term(term: str) -> str:
    """Normalize a single search term.
    
    Parameters
    ----------
    term : str
        A single-word term to normalize.
    
    Returns
    -------
    str
        The normalized (stemmed) form of the term.
    
    Raises
    ------
    ValueError
        If term normalizes to more or fewer than one token.
    """
    tokens = normalize_tokens(term, stopwords=stopwords_list, stem=True)
    if len(tokens) != 1:
        raise ValueError(f"term must be a single token, got {len(tokens)}: {tokens}")
    return tokens[0]

MatchType = Literal["exact", "partial", "index"]

def _words_matching_exact(query: str, items: list) -> list:
    result = []
    clean_query = preprocessing(query)
    for item in items:
        title = item.get("title", "")
        clean_title = preprocessing(title)
        if clean_query == clean_title:
            result.append(item)
    return result

def _words_matching_partial(query: str, items: list) -> list:
    result = []
    # preprocess the query (lowercase, remove punctuation) and tokenize it, then remove stopwords and stem the tokens
    tokenize_query = normalize_tokens(query, stopwords=stopwords_list, stem=True)
    for item in items:
        title = item.get("title", "")
        title_token = normalize_tokens(title, stopwords=stopwords_list, stem=True)
        for tq in tokenize_query:
            for tt in title_token:
                if tq in tt:
                    result.append(item)
    return result

def _words_matching_index(query: str, index_dict: dict, docmap_dict: dict) -> list:
    results = []
    seen = set()
    tokenize_query = normalize_tokens(query, stopwords=stopwords_list, stem=True)
    for token in tokenize_query:
        doc_ids = sorted(index_dict.get(token, set()))
        for doc_id in doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)

            doc = docmap_dict.get(doc_id)
            if doc is None:
                continue
            results.append(doc)

    return results[:5]

_MATCHERS = {
    "exact": _words_matching_exact,
    "partial": _words_matching_partial,
    "index": _words_matching_index
    }

def key_word_search(query: str, items: list, match_type: MatchType = "partial") -> list:
    try:
        matcher = _MATCHERS[match_type]
    except KeyError as exc:
        raise ValueError(
            f"Invalid match_type: {match_type}. Expected 'exact' or 'partial'. or 'index'."
        ) from exc
    return matcher(query, items)
    
def search_command(query_token: str, items: list, match_type: MatchType = "partial") -> list:
    if match_type == "index":
        invertedindex = InvertedIndex()
        index_dict, docmap_dict, _, _ = invertedindex.load()
        if not index_dict or not docmap_dict:
            raise ValueError("index_dict and docmap_dict must be provided for index matching.")
        return _words_matching_index(query_token, index_dict, docmap_dict)
    return key_word_search(query_token, items, match_type)

def build_index_command() -> None:
    invertedindex = InvertedIndex()
    invertedindex.build()
    invertedindex.save()
    # print example query result for "merida"
    # docs=invertedindex.get_documents("merida")
    # print(f"First document for token 'merida' = {docs[0]}")

def bm25_idf_command(term:str) -> float:
    invertedindex = InvertedIndex()
    invertedindex.load()
    return invertedindex.get_bm25_idf(term)

def bm25_tf_command(doc_id:int, term:str) -> float:
    invertedindex = InvertedIndex()
    invertedindex.load()
    return invertedindex.get_bm25_tf(doc_id, term)

def bm25_search_command(query:str, limit=5) -> list:
    invertedindex = InvertedIndex()
    invertedindex.load()
    return invertedindex.bm25_search(query, limit)

