import string
from nltk.stem import PorterStemmer

def preprocessing(text: str) -> str:
    """Normalize text by lowercasing and removing punctuation.

    Parameters
    ----------
    text : str
        Input text to normalize.

    Returns
    -------
    str
        Normalized text in lowercase with all ASCII punctuation characters
        (as defined by :data:`string.punctuation`) removed.

    Notes
    -----
    This function removes punctuation characters only. It does not perform
    stemming/lemmatization, stopword removal, or Unicode normalization.

    Examples
    --------
    >>> preprocessing("Hello, World!")
    'hello world'
    """
    text = text.lower()
    table = str.maketrans("", "", string.punctuation)
    clean_text = text.translate(table)
    return clean_text

def tokenize_text(text: str) -> list:
    """Tokenize text by splitting on whitespace.

    Parameters
    ----------
    text : str
        Input text to tokenize.

    Returns
    -------
    list
        List of tokens produced by ``str.split()``, which splits on any
        whitespace and collapses consecutive whitespace.

    Examples
    --------
    >>> tokenize_text("a  b\tc")
    ['a', 'b', 'c']
    """
    return text.split()

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