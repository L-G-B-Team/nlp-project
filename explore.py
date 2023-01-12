from typing import Union

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from wordcloud import WordCloud
from IPython.display import Markdown as md

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
    
def get_ngram_frequency(ser: pd.Series, n: int = 1) -> pd.Series:
    # TODO Docstring
    words = ' '.join(ser).split()
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
