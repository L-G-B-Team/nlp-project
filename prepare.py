import unicodedata

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from typing import List
import re
from nltk.corpus import stopwords as stpwrds

stopwords = stpwrds.words('english')

TOP_5_LANGUAGES = ['JavaScript', 'Python', 'TypeScript', 'Go', 'Java']


def basic_clean(string: str) -> str:
    '''
    Cleans string by converting to lower case and removing non-ACII characters
    # Parameters
    string: string to be cleaned
    # Returns
    cleaned string
    '''
    string = string.lower()
    string = unicodedata.normalize('NFKD', string).encode(
        'ascii', 'ignore').decode('utf-8', 'ignore')
    string = re.sub(r"[^a-z0-9\s]", '', string)
    return string


def tokenize(string: str) -> str:
    '''
    Tokenizes a given string
    # Parameters
    string: string to be tokenized
    # Returns
    tokenized string
    '''
    tok = nltk.tokenize.ToktokTokenizer()
    return tok.tokenize(string, return_str=True)


def stem(tokens: str) -> str:
    '''
    Stems given string
    # Parameters
    tokens: tokenized string to be stemmed
    # Returns
    stemmed string
    '''
    ps = nltk.porter.PorterStemmer()
    ret = [ps.stem(s) for s in tokens.split()]
    return ' '.join(ret)


def lemmatize(tokens: str) -> str:
    '''
    Lemmatizes given string
    # Parameters
    tokens: tokenized string to be lemmatized
    # Returns
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
    # Parameters
    tokenized: initial string

    extra_words: list of strings of additional stop words to remove

    exclude_words: list of strings of stop words to keep in strings
    # Returns
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
    # Parameters
    string: string to be cleaned

    extra_words: list of strings of additional stop words to remove

    exclude_words: list of strings of stop words to keep in strings
    # Returns
    The cleaned string
    '''
    string = basic_clean(string)
    string = tokenize(string)
    return remove_stopwords(string, extra_words, exclude_words)


def prep_df_for_nlp(df: pd.DataFrame, series_to_prep: str,
                    extra_words: List[str] = [],
                    exclude_words: List[str] = []) -> pd.DataFrame:
    '''
    Cleans and prepares a `DataFrame` for NLP,
    adds cleaned, stemmed, and lemmatized columns
    and collapses languages outside the top 5 to 'Other'
    # Parameters
        df: `DataFrame` to be cleaned

        series_to_prep: name of the series to be prepared within `df`

        extra_words: list of strings of additional stop words to remove

        exclude_words: list of strings of stop words to keep in strings

    # Returns
    Prepared `DataFrame` with additional columns containing cleaned data,
    stemmmed, and lemmatized data
    '''
    df.readme_contents = df.readme_contents.astype('str')
    # Clean data
    df['clean'] = df[series_to_prep].apply(
        squeaky_clean, exclude_words=exclude_words, extra_words=extra_words)
    # Stem cleaned data
    df['stem'] = df['clean'].apply(stem)
    # lemmatizes clean data
    df['lemmatized'] = df['clean'].apply(lemmatize)
    # change languages other than Top 5 languages to other
    language_mask = (~df.language.isin(TOP_5_LANGUAGES)
                     & ~df.language.isna())
    df.loc[language_mask, 'language'] = 'Other'
    # changes language to category
    df.language = df.language.astype('category')

    return df

############################################################ DIRECT CALLS FOR LANGUAGE SERIES

def series_generator(df):
    '''This function takes in the data frame from 
    prep_df_for_nlp and creates 6 pd.Series based on 
    the programing language. The series creates are 
    lists of strings that are the words in the READMEs'''
    
    # running input funtion through prep_df_for_nlp
    df = prep_df_for_nlp(df)
    
    # generates series for the top five languages
    JavaScript_words_series = (' '.join(df[df.language == 'JavaScript']['readme_contents']))
    Python_words_series = (' '.join(df[df.language == 'Python']['readme_contents']))
    TypeScript_words_series = (' '.join(df[df.language == 'TypeScript']['readme_contents']))
    Go_words_words_series = (' '.join(df[df.language == 'Go']['readme_contents']))
    Java_words_series = (' '.join(df[df.language == 'Java']['readme_contents']))

    # a series of words for all readme contents
    all_words_series = (' '.join(df['readme_contents']))






