import pathlib
import os
import sys
import string

def preprocessing(text:str) -> str:
    # make the text lower case
    text = text.lower()
    # create a translation table, (from string, to string, character dict to remove)
    table = str.maketrans("", "", string.punctuation)
    # apply the mapping
    clean_text = text.translate(table)
    return clean_text

def tokenize_text(text:str) -> list:
    # tokenization
    split_text = text.split(' ')
    split_text = list(filter(lambda x: len(x)!=0, split_text))
    return split_text