import pprint
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown as md
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import prepare as p

# order the lanugages
lang_order = ['JavaScript', 'TypeScript', 'Go',
              'Python', 'Java', 'Other', 'Not Listed']


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
    sns.heatmap(ngrams)


def language_distribution(df):
    # distribution of our repos by language
    fig = plt.figure(figsize=(20, 10))
    ax = plt.subplot(111)
    sns.countplot(data=df, x='language', ax=ax,
                  palette='colorblind', order=lang_order)
    plt.show()


def language_name_chi2(df, lang):
    series = df.language == lang
    has_word = df.lemmatized.str.contains(lang.lower())
    ctab = pd.crosstab(series, has_word)
    stat, p, degf, expected = stats.chi2_contingency(ctab)
    print(f'Chi^2 Stat:{stat}\np-value: {p}')
    return p_to_md(p)


def language_name_percentage_plot(df):
    # split into a df per language category
    go, java, javascript, not_listed, other, python, typescript = split_by_language(
        df)

    # get the word counts for each word in each readme by language
    javascript_words_freq = get_ngram_frequency(javascript.lemmatized)
    python_words_freq = get_ngram_frequency(python.lemmatized)
    typescript_words_freq = get_ngram_frequency(typescript.lemmatized)
    go_words_freq = get_ngram_frequency(go.lemmatized)
    other_series_freq = get_ngram_frequency(other.lemmatized)
    not_listed_freq = get_ngram_frequency(not_listed.lemmatized)
    java_words_freq = get_ngram_frequency(java.lemmatized)
    all_words_freq = get_ngram_frequency(df.lemmatized)

    # put all word counts together in a df
    word_counts = (pd.concat([all_words_freq, javascript_words_freq, typescript_words_freq, go_words_freq, python_words_freq, java_words_freq, other_series_freq, not_listed_freq], axis=1, sort=True)
                   .set_axis(['all', 'javascript', 'typescript', 'go', 'python', 'java', 'other', 'not_listed'], axis=1, inplace=False)
                   .fillna(0)
                   .apply(lambda s: s.astype(int)))

    # limit to only the word counts of the names of programming languages
    word_counts_limited = word_counts[(word_counts.index == 'javascript') | (word_counts.index == 'python') | (
        word_counts.index == 'typescript') | (word_counts.index == 'go') | (word_counts.index == 'java')]

    # plot the percentage of how many word count frequency
    fig = plt.figure(figsize=(20, 10))
    ax = plt.subplot(111)
    plt.rcParams.update({'font.size': 12})

    (word_counts_limited.sort_values('all', ascending=False)
     .head(20)
     .apply(lambda row: row/row['all'], axis=1)
     .drop(columns='all')
     .sort_values(by='javascript')
     .plot.barh(stacked=True, width=1, ec='k', legend=False, ax=ax, color=sns.color_palette('colorblind'))
     )
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=7, fancybox=True, shadow=True)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .5, 1], ['0%', '50%', '100%'])
    plt.show()


def readme_len_plot(df):
    fig = plt.figure(figsize=(20, 10))
    ax = plt.subplot(111)
    sns.barplot(x="language", y="lemmatized_len", data=df,
                palette='colorblind', ax=ax, order=lang_order)
    plt.show()


def readme_len_kruskal(df):
    # split into a df per language category
    go, java, javascript, not_listed, other, python, typescript = split_by_language(
        df)

    stat, p = stats.kruskal(go.lemmatized_len, java.lemmatized_len, javascript.lemmatized_len,
                            not_listed.lemmatized_len, other.lemmatized_len, python.lemmatized_len, typescript.lemmatized_len)
    return p_to_md(p)


def title_chi2(df, word):
    lang = df.language
    has_word = df.repo.str.contains(word)
    ctab = pd.crosstab(lang, has_word)
    stat, p, degf, expected = stats.chi2_contingency(ctab)
    print(f'Chi^2 Stat:{stat}\np-value: {p}')
    return p_to_md(p)


def get_idf(df):
    tfidf = TfidfVectorizer()
    bag_of_words = tfidf.fit_transform(df.lemmatized)
    pd.DataFrame(bag_of_words.todense(),
                 columns=tfidf.get_feature_names_out())
    idf_values = pd.Series(
        dict(
            zip(
                tfidf.get_feature_names_out(), tfidf.idf_)))
    return idf_values.describe()


def percentage_of_language_per_word(df):
    go, java, javascript, not_listed, other, python, typescript = split_by_language(
        df)
    javascript_title_freq = get_ngram_frequency(javascript.repo)
    python_title_freq = get_ngram_frequency(python.repo)
    typescript_title_freq = get_ngram_frequency(typescript.repo)
    go_title_freq = get_ngram_frequency(go.repo)
    other_series_freq = get_ngram_frequency(other.repo)
    not_listed_freq = get_ngram_frequency(not_listed.repo)
    java_title_freq = get_ngram_frequency(java.repo)
    all_title_freq = get_ngram_frequency(df.repo)

    title_word_counts = (pd.concat([all_title_freq, javascript_title_freq, typescript_title_freq, go_title_freq, python_title_freq, java_title_freq, other_series_freq, not_listed_freq], axis=1, sort=True)
                         .set_axis(['all', 'javascript', 'typescript', 'go', 'python', 'java', 'other', 'not_listed'], axis=1, inplace=False)
                         .fillna(0)
                         .apply(lambda s: s.astype(int)))
    title_word_counts_limited = title_word_counts[(title_word_counts.index == 'awesome') | (
        title_word_counts.index == 'react') | (title_word_counts.index == 'go')]

    return title_word_counts_limited


def significant_words_graph(train: pd.DataFrame) -> None:
    significant_words = ['awesome', 'go', 'react']
    ret_df = pd.DataFrame()
    for word in significant_words:
        ret_df[word] = train.groupby('language').lemmatized.agg(
            lambda l: l.str.contains(word).sum())
    sns.barplot(data=ret_df,x=)
