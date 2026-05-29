import os
import json
import string
from nltk.stem import PorterStemmer
from functools import lru_cache

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
STOP_WORDS = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
MOVIE_DATA = os.path.join(PROJECT_ROOT, "data", "movies.json")

@lru_cache(maxsize=1)
def get_stopwords() -> list[str]:
    with open(STOP_WORDS) as f:
        return [preprocessing(word) for word in f.read().splitlines()]
    
@lru_cache(maxsize=1)
def get_movies() -> list[dict]:
    with open(MOVIE_DATA, "r", encoding="utf-8") as f:
        movies = json.load(f)
        return movies['movies']
    
def read_json(file_path: str | os.PathLike[str]) -> list:
    '''Read a JSON file and return the list of movies contained in it.
    Parameters
    ----------
    file_path : str | os.PathLike[str]
        Path to the JSON file containing movie data. The file is expected to
        have a structure like {"movies": [ ... ]}, where the value associated with
        the "movies" key is a list of movie objects.
    Returns
    -------
    list
        A list of movie objects extracted from the JSON file. Each movie object is
        expected to be a dictionary with keys such as "id", "title", and "description".
    '''
    
    with open(file_path, "r", encoding="utf-8") as f:
        movies = json.load(f)
        movie_list = movies['movies']
        return movie_list
    
def preprocessing(text: str) -> str:
    '''Preprocess the input text by converting to lowercase and removing punctuation.
    Parameters
    ----------
    text : str
        The input text to preprocess.
    Returns
    -------
    str
        The preprocessed text, which is the original text converted to lowercase and with all punctuation removed.
    '''
    def change_to_lowercase(text: str) -> str:
        return text.lower()
    def remove_punctuation(text: str) -> str:
        table = str.maketrans("", "", string.punctuation)
        clean_text = text.translate(table)
        return clean_text
    if text is None:
        return ''
    text = change_to_lowercase(text)
    text = remove_punctuation(text)
    return text

def tokenize_text(text: str) -> list:
    '''Tokenize the input text by splitting on whitespace and remove empty token.
    '''
    valid_tokens = []
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:                        
            valid_tokens.append(token)
    return valid_tokens
        
def stemming(text: list) -> list:
    """Stem tokens using NLTK's Porter stemmer.

    Parameters
    ----------
    text : list
        Sequence of token strings to stem.

    Returns
    -------
    list
        Stemmed tokens, in the same order as the input.

    See Also
    --------
    nltk.stem.PorterStemmer : The stemmer implementation used.

    Examples
    --------
    >>> stemming(["running", "runs", "ran"])
    ['run', 'run', 'ran']
    """
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in text:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words

def load_stopwords(file_path: str) -> list:
    '''Remove stopwords'''
    with open(file_path) as f:
        stopwords = f.read().splitlines()
        return stopwords

def normalize_tokens(text: str, stopwords: list[str] | None = None, stem: bool = True) -> list[str]:
    """Preprocess text into normalized tokens.

    Parameters
    ----------
    text : str
        Input text to normalize.
    stopwords : list[str], optional
        List of stopword strings to filter out. If None, no stopword removal.
    stem : bool, default True
        If True, apply stemming to tokens.

    Returns
    -------
    list[str]
        List of normalized tokens.
    """
    clean = preprocessing(text)
    tokens = tokenize_text(clean)
    if stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    if stem:
        tokens = stemming(tokens)
    return tokens
