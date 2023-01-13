from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown as md
from wordcloud import WordCloud

import prepare as p


def p_to_md(p: float, alpha: float = .05, **kwargs) -> md:
    '''
    returns the result of a p test as a `Markdown` object
    ## Parameters
    p: `float` of the p value from performed Hypothesis test
    alpha: `float` of alpha value for test, defaults to 0.05
    kwargs: any additional return values of statistical test
    ## Returns
    formatted `Markdown` object containing results of hypothesis test.

    '''
    ret_str = ''
    p_flag = p < alpha
    for k, v in kwargs.items():
        ret_str += f'## {k} = {v}\n\n'
    ret_str += f'## Because $\\alpha$ {">" if p_flag else "<"} p,' + \
        f'we {"failed to " if ~(p_flag) else ""} reject $H_0$'
    return md(ret_str)


def t_to_md_1samp(p: float, t: float, alpha: float = .05, **kwargs):
    '''takes a p-value, alpha, and any T-test arguments and
    creates a Markdown object with the information.
    ## Parameters
    p: float of the p value from run T-Test
    t: float of the t-value from run T-Test
    alpha: desired alpha value, defaults to 0.05
    ## Returns
    `IPython.display.Markdown` object with results of the statistical test
    '''
    ret_str = ''
    t_flag = t > 0
    p_flag = p/2 < alpha
    ret_str += f'## t = {t} \n\n'
    for k, v in kwargs.items():
        ret_str += f'## {k} = {v}\n\n'
    ret_str += f' ## p/2 = {p/2} \n\n'
    ret_str += (f'## Because t {">" if t_flag else "<"} 0 '
                f'and $\\alpha$ {">" if p_flag else "<"} p/2, '
                f'we {"failed to " if ~(t_flag & p_flag) else ""} '
                ' reject $H_0$')
    return md(ret_str)


def t_to_md(p: float, t: float, alpha: float = .05, **kwargs):
    '''takes a p-value, alpha, and any T-test arguments and
    creates a Markdown object with the information.
    ## Parameters
    p: float of the p value from run T-Test
    t: float of the t-value from run T-Test
    alpha: desired alpha value, defaults to 0.05
    ## Returns
    `IPython.display.Markdown` object with results of the statistical test
    '''
    ret_str = ''
    t_flag = t > 0
    p_flag = p < alpha
    ret_str += f'## t = {t} \n\n'
    for k, v in kwargs.items():
        ret_str += f'## {k} = {v}\n\n'
    ret_str += f' ## p = {p} \n\n'
    ret_str += (f'## Because t {">" if t_flag else "<"} 0 '
                f'and $\\alpha$ {">" if p_flag else "<"} p, '
                f'we {"failed to " if ~(t_flag & p_flag) else ""} '
                ' reject $H_0$')
    return md(ret_str)


def get_ngram_frequency(ser: Union[pd.Series, str], n: int = 1) -> pd.Series:
    '''
    Generates a series of the frequency of occurences
    of ngrams in provided documents
    ## Parameters
    ser: either a `Series` or `str containing the documents
    n: no of words to use in ngrams. Default value of 1
    ## Returns
    a `Series` showing the value counts of each ngram
    '''
    if isinstance(ser, pd.Series):
        words = ' '.join(ser).split()
    else:
        words = ser.split()
    if n > 1:
        ngrams = nltk.ngrams(words, n)
        words = [' '.join(n) for n in ngrams]
    return pd.Series(words).value_counts()


def generate_word_cloud(ser: pd.Series, ngram: int = 1,
                        ax: Union[plt.Axes, None] = None,
                        **kwargs) -> Union[plt.Axes, None]:
    # TODO Docstring
    if ser.dtype != np.int64:
        ser = get_ngram_frequency(ser, ngram)
    wc = WordCloud(**kwargs).generate_from_frequencies(ser.to_dict())
    if ax is not None:
        ax.imshow(wc)
        return ax
    plt.imshow(wc)
    plt.show()


def get_word_frequency(readme: str) -> pd.Series:
    # TODO Docstring
    val_counts = pd.Series(readme.split()).value_counts()[:5]
    ret_ser = pd.Series()
    for index, word, count in zip(range(1, 6), val_counts.index, val_counts):
        ret_ser[f'word_{index}'] = word
        ret_ser[f'count_{index}'] = count
    return ret_ser


def top_five_words(series: pd.Series) -> pd.DataFrame:
    # TODO Docstring
    readme_counts = series.apply(get_word_frequency)
    for i in range(1, 6):
        readme_counts[f'count_{i}'].fillna(0)
        readme_counts[f'count_{i}'] = readme_counts[f'count_{i}'
                                                    ].fillna(0).astype('int')
    return readme_counts


def split_by_language(df):
    '''
    Takes in the dataframe and splits on the languages
    Returns seven dataframes, one per language
    '''
    go = df[df.language == 'Go']
    java = df[df.language == 'Java']
    javascript = df[df.language == 'JavaScript']
    not_listed = df[df.language == 'Not Listed']
    other = df[df.language == 'Other']
    python = df[df.language == 'Python']
    typescript = df[df.language == 'TypeScript']

    return go, java, javascript, not_listed, other, python, typescript


def top_ngrams_by_group(df: pd.DataFrame, top_n: int = 10, n: int = 1,
                        content: str = 'lemmatized',
                        separator: str = 'language') -> pd.DataFrame:
    # TODO Docstring
    top_ten = pd.Series()
    if n > 1:
        top_ten = get_ngram_frequency(df[content], n)
        top_ten = top_ten[top_ten.index.str.len() > 3][:top_n]
    else:
        words = ' '.join(df[content].to_list())
        top_ten = pd.Series(words.split()).value_counts()
        top_ten = top_ten[top_ten.index.str.len() > 3][:top_n]
    percentage_lst = []
    for s in df[separator].unique():
        ser = pd.Series(name=s)
        separator_group = df[df[separator] == s]
        for w in top_ten.index:
            ser[w] = separator_group[
                separator_group[
                    content].str.contains(w)
            ].shape[0] / separator_group.shape[0]
        percentage_lst.append(ser)
    return pd.concat(percentage_lst, axis=1)


def word_heat_map(df: pd.DataFrame, top_n: int = 10, n: int = 1) -> None:
    # TODO Docstring
    ngrams = top_ngrams_by_group(df, top_n, n)
    sns.heatmap(ngrams,cmap='rainbow_r')
