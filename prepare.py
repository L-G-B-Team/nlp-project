import unicodedata

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from typing import List
import re
from nltk.corpus import stopwords as stpwrds

from sklearn.model_selection import train_test_split

stopwords = stpwrds.words('english')

LANGUAGE_COUNT = 5

EXTRA_WORDS = ['&#9;',
               'example',
               'api',
               'data',
               'return',
               'support',
               'new',
               'image',
               'install',
               'version',
               'img',
               'build',
               'object',
               'open',
               'user',
               '3',
               'td',
               'web',
               'docker',
               'opensource',
               'server',
               'option',
               'type',
               'tool',
               'test',
               'method',
               'list',
               'name',
               'number',
               'import',
               'string']


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
    top_n_languages = df.language.value_counts(
    )[:LANGUAGE_COUNT].index.to_list()
    language_mask = (~df.language.isin(top_n_languages)
                     & ~df.language.isna())
    df.loc[language_mask, 'language'] = 'Other'
    df.loc[df.language.isna(), 'language'] = 'Not Listed'
    # changes language to category
    df.language = df.language.astype('category')
    # add character length of lemmatized content
    df['lemmatized_len'] = df.lemmatized.apply(lambda s: len(s))

    return df

# DIRECT CALLS FOR LANGUAGE SERIES


def series_generator(df):
    '''This function takes in the data frame from 
    prep_df_for_nlp and creates 6 pd.Series based on 
    the programing language. The series creates are 
    lists of strings that are the words in the READMEs'''

    # generates series for the top five languages
    javascript_words_series = (
        ' '.join(df[df.language == 'JavaScript']['readme_contents']))
    python_words_series = (
        ' '.join(df[df.language == 'Python']['readme_contents']))
    typescript_words_series = (
        ' '.join(df[df.language == 'TypeScript']['readme_contents']))
    go_words_series = (' '.join(df[df.language == 'Go']['readme_contents']))
    java_words_series = (
        ' '.join(df[df.language == 'Java']['readme_contents']))

    # generates series for the languages not listed and not in the top 5
    language_not_listed_series = (
        ' '.join(df[df.language == 'Not Listed']['readme_contents']))
    other_series = (' '.join(df[df.language == 'Other']['readme_contents']))

    # a series of words for all readme contents
    all_words_series = (' '.join(df['readme_contents']))

    # returned in order of: javascript_series, python_series, type_series, go_series, java_series, unlisted, other, all_words_series
    return (javascript_words_series, python_words_series, typescript_words_series, go_words_series, java_words_series, language_not_listed_series, other_series, all_words_series)


def split_data(df, target, test_size=0.15):
    '''
    Takes in a data frame and the train size
    It returns train, validate , and test data frames
    with validate being 0.05 bigger than test and train has the rest of the data.
    '''
    train, test = train_test_split(df, stratify=df[target], test_size = test_size , random_state=27)
    train, validate = train_test_split(train,  stratify=train[target], test_size = (test_size + 0.05)/(1-test_size), random_state=27)
    
    return train, validate, test
