import numpy as np
import pandas as pd
from typing import Union, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score

ModelType = Union[LogisticRegression, DecisionTreeClassifier,
                  RandomForestClassifier, KNeighborsClassifier]


def model_data(model: ModelType, features: pd.DataFrame, target: Union[pd.Series, None] = None, result_suffix: str = '') -> pd.DataFrame:
    # TODO Docstring
    y_hat = pd.Series()
    try:
        y_hat = pd.Series(model.predict(features))
    except NotFittedError:
        if target is None:
            raise NotFittedError('Model not fit and target not provided')
        model.fit(features, target)
        y_hat = pd.Series(model.predict(features))
    y_hat.name = (result_suffix if len(result_suffix) > 0 else '')
    y_hat.index = target.index
    return y_hat


def tune_decision_tree(features: pd.DataFrame, target: pd.Series,
                       max_depth: Tuple[int, int, int] = (2, 31, 1),
                       metric: callable = accuracy_score):
    # TODO
    ret_lst = {}
    for depth in range(max_depth[0], max_depth[1], max_depth[2]):
        model = RandomForestClassifier(max_depth=depth, random_state=27)
        yhat = model_data(model, features, target, 'max_depth_'+str(depth))
        ret_lst[yhat.name] = metric(target.to_numpy(), yhat.to_numpy())
    return pd.Series(ret_lst, name='Decision Tree')


def tune_random_forest(features: pd.DataFrame, target: pd.Series,
                       max_depth: Tuple[int, int, int] = (2, 31, 1),
                       min_samples_leaf: Tuple[int, int, int] = (2, 31, 1),
                       metric: callable = accuracy_score) -> pd.Series:
    ret_ser = {}
    for depth in range(max_depth[0], max_depth[1], max_depth[2]):
        ret_sub_ser = {}

        for leaf in range(min_samples_leaf[0],
                          min_samples_leaf[1],
                          min_samples_leaf[2]):
            model = RandomForestClassifier(
                max_depth=depth, min_samples_leaf=leaf)
            yhat = model_data(model, features, target,
                              result_suffix='min_samples_leaf_' + str(leaf))
            ret_sub_ser[yhat.name] = accuracy_score(target.to_numpy(),
                                                    yhat.to_numpy())
        ret_ser['max_depth_' + str(depth)] = ret_sub_ser
    return pd.DataFrame(ret_ser)
