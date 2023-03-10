import unicodedata

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union
import re
from nltk.corpus import stopwords as stpwrds

from sklearn.model_selection import train_test_split

stopwords = stpwrds.words('english')

LANGUAGE_COUNT = 5

EXTRA_WORDS = ['&#9;',
               '3',
               'api',
               'build',
               'data',
               'docker',
               'example',
               'image',
               'img',
               'import',
               'install',
               'list',
               'method',
               'name',
               'new',
               'number',
               'object',
               'open',
               'opensource',
               'option',
               'return',
               'server',
               'string',
               'support',
               'td',
               'test',
               'tool',
               'type',
               'user',
               'version',
               'web']


def basic_clean(input_str: str) -> str:
    '''
    Cleans string by converting to lower case and removing non-ACII characters
    # Parameters
    string: string to be cleaned
    # Returns
    cleaned string
    '''
    input_str = input_str.lower()
    input_str = unicodedata.normalize('NFKD', input_str).encode(
        'ascii', 'ignore').decode('utf-8', 'ignore')
    input_str = re.sub(r'\<([^\s])+\s+.*\>([^\<])?\<\1\>', '', input_str)
    input_str = re.sub(r'https\:\/\/[^\s]', '', input_str)
    input_str = re.sub(r"[^a-z0-9\s]", '', input_str)
    input_str = re.sub(r'\s*\.\s+', '', input_str)
    return input_str


def tokenize(input_str: str) -> str:
    '''
    Tokenizes a given string
    # Parameters
    string: string to be tokenized
    # Returns
    tokenized string
    '''
    tok = nltk.tokenize.ToktokTokenizer()
    return tok.tokenize(input_str, return_str=True)


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


def remove_links_and_html(input_str: str) -> str:
    input_str = re.sub(
        r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', '', input_str)
    input_str = re.sub(r'\s*http(s)?://([^\s])+', '', input_str)
    input_str = re.sub(r'/', ' ', input_str)
    return re.sub(r'\[([^\]]+)\]\(.*\)', r'\1', input_str)


def squeaky_clean(input_str: str, extra_words: List[str] = [],
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
    input_str = remove_links_and_html(input_str)
    input_str = basic_clean(input_str)
    input_str = tokenize(input_str)
    return remove_stopwords(input_str, extra_words, exclude_words)


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
        squeaky_clean, exclude_words=exclude_words, extra_words=EXTRA_WORDS)
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
    split_series = df.repo.apply(lambda s: s.split('/'))
    df['username'] = split_series.apply(lambda arr: arr[0])
    df.repo = split_series.apply(lambda arr: arr[1])
    df.repo = df.repo.apply(lambda s: s.replace('-', ' '))
    # changes language to category
    df.language = df.language.astype('category')
    # add character length of lemmatized content
    df['lemmatized_len'] = df.lemmatized.apply(lambda s: len(s))
    return df

# DIRECT CALLS FOR LANGUAGE SERIES


def series_generator(df: pd.DataFrame) -> Tuple[str, str, str, str,
                                                str, str, str]:
    '''This function takes in the data frame from
    prep_df_for_nlp and creates 6 pd.Series based on
    the programing language. The series creates are
    lists of strings that are the words in the READMEs'''

    # generates series for the top five languages
    series_dict = generate_series(df.lemmatized, df.language)
    javascript_words_series = series_dict['JavaScript']
    python_words_series = series_dict['Python']
    typescript_words_series = series_dict['TypeScript']
    go_words_series = series_dict['Go']
    other_series = series_dict['Other']
    language_not_listed_series = series_dict['Not Listed']
    java_words_series = series_dict['Java']
    all_words_series = series_dict['All']

    # returned in order of: javascript_series, python_series, type_series, go_series, java_series,unlisted, other, all_words_series
    return (javascript_words_series, python_words_series,
            typescript_words_series, go_words_series,
            java_words_series, language_not_listed_series, other_series,
            all_words_series)


def generate_series(content: pd.Series, separator: Union[pd.Series, None] = None) -> Union[Dict[str, str], str]:
    # TODO Docstring
    ret_dict = {}
    if separator is None:
        return ' '.join(content.to_list())
    for s in separator.unique():
        indices = separator[separator == s].index
        ret_dict[s] = ' '.join(content.iloc[indices].to_list())
    ret_dict['All'] = ' '.join(content.to_list())

    return ret_dict


def split_data(df: pd.DataFrame, target: str, test_size: float = 0.15):
    '''
    Takes in a data frame and the train size
    It returns train, validate , and test data frames
    with validate being 0.05 bigger than test and train has the rest of the data.
    '''
    train, test = train_test_split(
        df, stratify=df[target], test_size=test_size, random_state=27)
    train, validate = train_test_split(train,  stratify=train[target],
                                       test_size=(
        test_size + 0.05)/(1-test_size), random_state=27)

    return train, validate, test

def prepare_readme(readme:str)->str:
    #TODO Docstring
    readme = squeaky_clean(readme)
    return lemmatize(readme)