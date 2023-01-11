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
    # TODO Docstring
    string = string.lower()
    string = unicodedata.normalize('NFKD', string).encode(
        'ascii', 'ignore').decode('utf-8', 'ignore')
    string = re.sub(r"[^a-z0-9'\s]", '', string)
    return string


def tokenize(string: str) -> str:
    # TODO Docstring
    tok = nltk.tokenize.ToktokTokenizer()
    return tok.tokenize(string, return_str=True)


def stem(tokens: str) -> str:
    # TODO Docstring
    ps = nltk.porter.PorterStemmer()
    ret = [ps.stem(s) for s in tokens.split()]
    return ' '.join(ret)


def lemmatize(tokens: str) -> str:
    # TODO Docstring
    lem = nltk.stem.WordNetLemmatizer()
    ret = [lem.lemmatize(s) for s in tokens.split()]
    return ' '.join(ret)


def remove_stopwords(tokens: str,
                     extra_words: List[str] = [],
                     exclude_words: List[str] = []) -> str:
    # TODO Docstring
    tokens = [t for t in tokens.split()]
    for exc in exclude_words:
        stopwords.remove(exc)
    for ext in extra_words:
        stopwords.append(ext)
    stopped = [t for t in tokens if t not in stopwords]
    return ' '.join(stopped)


def squeaky_clean(string: str, extra_words: List[str] = [], exclude_words: List[str] = []) -> str:
    string = basic_clean(string)
    string = tokenize(string)
    return remove_stopwords(string, extra_words, exclude_words)


def prep_df_for_nlp(df: pd.DataFrame, ser: str,
                    extra_words: List[str] = [],
                    exclude_words: List[str] = []) -> pd.DataFrame:
    df['clean'] = df[ser].apply(
        squeaky_clean, exclude_words=exclude_words, extra_words=extra_words)
    df['stem'] = df['clean'].apply(stem)
    df['lemmatized'] = df['clean'].apply(lemmatize)
    return df
