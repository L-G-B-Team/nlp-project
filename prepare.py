import unicodedata

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from typing import List
import re
from nltk.corpus import stopwords as stpwrds

stopwords = stpwrds.words('english')


def basic_clean(string: str) -> str:
    '''
    Cleans string by converting to lower case and removing non-ACII characters
    ## Parameters
    string: string to be cleaned
    ## Returns
    cleaned string
    '''
    string = string.lower()
    string = unicodedata.normalize('NFKD', string).encode(
        'ascii', 'ignore').decode('utf-8', 'ignore')
    string = re.sub(r"[^a-z0-9'\s]", '', string)
    return string


def tokenize(string: str) -> str:
    '''
    Tokenizes a given string
    ## Parameters
    string: string to be tokenized
    ## Returns
    tokenized string
    '''
    tok = nltk.tokenize.ToktokTokenizer()
    return tok.tokenize(string, return_str=True)


def stem(tokens: str) -> str:
    '''
    Stems given string
    ## Parameters
    tokens: tokenized string to be stemmed
    ## Returns
    stemmed string
    '''
    ps = nltk.porter.PorterStemmer()
    ret = [ps.stem(s) for s in tokens.split()]
    return ' '.join(ret)


def lemmatize(tokens: str) -> str:
    '''
    Lemmatizes given string
    ## Parameters
    tokens: tokenized string to be lemmatized
    ## Returns
    lemmatized string
    '''
    lem = nltk.stem.WordNetLemmatizer()
    ret = [lem.lemmatize(s) for s in tokens.split()]
    return ' '.join(ret)


def remove_stopwords(tokens: str,
                     extra_words: List[str] = [],
                     exclude_words: List[str] = []) -> str:
    '''
    Removes stop words from string
    ## Parameters
    tokenized: initial string

    extra_words: list of strings of additional stop words to remove

    exclude_words: list of strings of stop words to keep in strings
    ## Returns
    string with stopwords removed
    '''
    tokens = [t for t in tokens.split()]
    for exc in exclude_words:
        stopwords.remove(exc)
    for ext in extra_words:
        stopwords.append(ext)
    stopped = [t for t in tokens if t not in stopwords]
    return ' '.join(stopped)


def squeaky_clean(string: str, extra_words: List[str] = [],
                  exclude_words: List[str] = []) -> str:
    '''
    cleans, tokenizes, and removes stop words from string
    ## Parameters
    string: string to be cleaned

    extra_words: list of strings of additional stop words to remove

    exclude_words: list of strings of stop words to keep in strings
    ## Returns
    The cleaned string
    '''
    string = basic_clean(string)
    string = tokenize(string)
    return remove_stopwords(string, extra_words, exclude_words)


def prep_df_for_nlp(df: pd.DataFrame, series_to_prep: str,
                    extra_words: List[str] = [],
                    exclude_words: List[str] = []) -> pd.DataFrame:
    '''
    Cleans and prepares a `DataFrame` for NLP
    ## Parameters
        df: `DataFrame` to be cleaned

        series_to_prep: name of the series to be prepared within `df`

        extra_words: list of strings of additional stop words to remove

        exclude_words: list of strings of stop words to keep in strings

    ## Returns
    Prepared `DataFrame`
    '''
    df['clean'] = df[series_to_prep].apply(
        squeaky_clean, exclude_words=exclude_words, extra_words=extra_words)
    df['stem'] = df['clean'].apply(stem)
    df['lemmatized'] = df['clean'].apply(lemmatize)
    return df


