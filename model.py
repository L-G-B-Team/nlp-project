import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score

ModelType = Union[LogisticRegression, DecisionTreeClassifier,
                  RandomForestClassifier, KNeighborsClassifier,
                  GradientBoostingClassifier]


def model_data(model: ModelType, features: pd.DataFrame, target: Union[pd.Series, None] = None, result_suffix: str = '') -> pd.DataFrame:
    # TODO Docstring
    y_hat = pd.Series()
    try:
        # gets predictions if model has been fitted
        y_hat = pd.Series(model.predict(features))
    # if model is not fitted, fits model and returns predictions
    except NotFittedError:
        if target is None:
            raise NotFittedError('Model not fit and target not provided')
        model.fit(features, target)
        y_hat = pd.Series(model.predict(features))
    y_hat.name = (result_suffix if len(result_suffix) > 0 else '')
    # changes indexes to match that of target
    y_hat.index = target.index
    return y_hat


def tune_decision_tree(train_features: pd.DataFrame, train_target: pd.Series,
                       valid_features: pd.DataFrame, valid_target: pd.Series,
                       max_depth: Tuple[int, int, int] = (2, 31, 1),

                       metric: callable = accuracy_score):
    '''

    ## Parameters
    features: `DataFrame` of features to model on
    target: `Series` of target variable
    max_depth: a tuple representing, in order, the minimum value for max_depth,
    the maximum value for max_depth, and the number of steps between.
    These function as parameters for a range.
    metric: a function for the metric used to evaluate the inputs
    ## Returns
    a `Series` containing the metric scores at each max_depth
    '''
    train_lst = {}
    valid_lst = {}
    for depth in range(max_depth[0], max_depth[1], max_depth[2]):
        model = RandomForestClassifier(max_depth=depth, random_state=27)
        train_yhat = model_data(model, train_features,
                                train_target, str(depth))
        valid_yhat = model_data(model, valid_features,
                                valid_target, str(depth))
        train_lst[train_yhat.name] = metric(
            train_target.to_numpy(), train_yhat.to_numpy())
        valid_lst[valid_yhat.name] = metric(
            valid_target.to_numpy(), valid_yhat.to_numpy())
    return pd.Series(train_lst, name='Train Data'), pd.Series(valid_lst, name='Valid Data')


def tune_random_forest(
        train_features: pd.DataFrame,
        train_target: pd.DataFrame,
        valid_features: pd.DataFrame, valid_target: pd.Series,
        max_depth: Tuple[int, int, int] = (2, 21, 1),
        min_samples_leaf: Tuple[int, int, int] = (2, 31, 1),
        metric: callable = accuracy_score) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Runs multiple versions of RandomForestClassifier
    with different hyperparameters to determine the most effective
    parameters for modeling.
    ## Parameters
    features: `DataFrame` of features to model on
    target: `Series` of target variable
    max_depth: a tuple representing, in order, the minimum value for max_depth,
    the maximum value for max_depth, and the number of steps between.
    These function as parameters for a range
    min_samples_leaf: a tuple of integers representing, in order,
    the minimum value for min_samples_leaf,
    the maximum value for min_samples_leaf, and the number of steps between.
    These are used as parameters for a range
    metric: a function for the metric used to evaluate the inputs
    ## Returns
    a `DataFrame` with the metric scores at each max_depth and min_samples_leaf
    '''
    train_ser = {}
    valid_ser = {}
    for depth in range(max_depth[0], max_depth[1], max_depth[2]):
        train_sub_ser = {}
        valid_sub_ser = {}
        for leaf in range(min_samples_leaf[0],
                          min_samples_leaf[1],
                          min_samples_leaf[2]):
            model = RandomForestClassifier(
                n_estimators=180, max_depth=depth, min_samples_leaf=leaf)
            train_yhat = model_data(model, train_features, train_target,
                                    result_suffix='min_samples_leaf_' + str(leaf))
            valid_yhat = model_data(model, valid_features, valid_target,
                                    result_suffix='min_samples_leaf_' + str(leaf))
            train_sub_ser[train_yhat.name] = metric(train_target.to_numpy(),
                                                    train_yhat.to_numpy())
            valid_sub_ser[valid_yhat.name] = metric(valid_target.to_numpy(),
                                                    valid_yhat.to_numpy())
        train_ser['n_estimators_' + str(depth)] = train_sub_ser
        valid_ser['n_estimators_' + str(depth)] = valid_sub_ser
    return (pd.DataFrame(train_ser),
            pd.DataFrame(valid_ser))


def scale(features: pd.Series, scaler: MinMaxScaler) -> pd.Series:
    indexes = features.index
    features = features.values.reshape(-1, 1)
    try:
        ret_series = scaler.transform(features)
    except NotFittedError as e:
        scaler = scaler.fit(features)
        ret_series = scaler.transform(features)
    return pd.DataFrame(ret_series, index=indexes, columns=['scaled_lemmatized_length'])


def encode_has_language(df):
    '''
    Takes in df and returns it with added features
    '''
    ret_df = pd.DataFrame()
    ret_df['has_java'] = df.lemmatized.str.contains('java')
    ret_df['has_javascript'] = df.lemmatized.str.contains('javascript')
    ret_df['has_python'] = df.lemmatized.str.contains('python')
    ret_df['has_typescript'] = df.lemmatized.str.contains('typescript')
    ret_df['has_awesome'] = df.repo.str.contains('awesome')
    ret_df['has_react'] = df.repo.str.contains('react')
    ret_df['has_go'] = df.repo.str.contains('go')
    ret_df.index = df.index

    return ret_df


def encode_for_model(train, validate, test):
    '''
    Takes in train, validate, test and adds the added features for all of them
    '''
    train = encode_has_language(train)
    validate = encode_has_language(validate)
    test = encode_has_language(test)
    return train, validate, test

# TODO XG BOOST


def tune_gradient_boost(
        train_features: pd.DataFrame,
        train_target: pd.DataFrame,
        valid_features: pd.DataFrame, valid_target: pd.Series,
        max_depth: Tuple[int, int, int] = (2, 21, 1),
        min_samples_leaf: Tuple[int, int, int] = (2, 31, 1),
        metric: callable = accuracy_score) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Runs multiple versions of GradientBoostingClassifier
    with different hyperparameters to determine the most effective
    parameters for modeling.
    ## Parameters
    features: `DataFrame` of features to model on
    target: `Series` of target variable
    max_depth: a tuple representing, in order, the minimum value for max_depth,
    the maximum value for max_depth, and the number of steps between.
    These function as parameters for a range
    min_samples_leaf: a tuple of integers representing, in order,
    the minimum value for min_samples_leaf,
    the maximum value for min_samples_leaf, and the number of steps between.
    These are used as parameters for a range
    metric: a function for the metric used to evaluate the inputs
    ## Returns
    a `DataFrame` with the metric scores at each max_depth and min_samples_leaf
    '''
    train_ser = {}
    valid_ser = {}
    for depth in range(max_depth[0], max_depth[1], max_depth[2]):
        train_sub_ser = {}
        valid_sub_ser = {}
        for leaf in range(min_samples_leaf[0],
                          min_samples_leaf[1],
                          min_samples_leaf[2]):
            model = GradientBoostingClassifier(
                n_estimators=100, max_depth=depth, min_samples_leaf=leaf)
            train_yhat = model_data(model, train_features, train_target,
                                    result_suffix=str(leaf))
            valid_yhat = model_data(model, valid_features, valid_target,
                                    result_suffix=str(leaf))
            train_sub_ser[train_yhat.name] = metric(train_target.to_numpy(),
                                                    train_yhat.to_numpy())
            valid_sub_ser[valid_yhat.name] = metric(valid_target.to_numpy(),
                                                    valid_yhat.to_numpy())
        train_ser[str(depth)] = train_sub_ser
        valid_ser[str(depth)] = valid_sub_ser
    return (pd.DataFrame(train_ser),
            pd.DataFrame(valid_ser))


def assess_model_performance(train: pd.DataFrame, validate: pd.DataFrame) -> plt.Axes:
    # TODO doctring

    # encode_has_language
    train_x = encode_has_language(train)
    valid_x = encode_has_language(validate)
    # scale lemmatized_len
    scaler = MinMaxScaler()
    scaled_valid = scale(validate.lemmatized_len, scaler)
    scaled_train = scale(train.lemmatized_len, scaler)
    # concat scaled_train and encoded_has_language
    train_x = pd.concat([scaled_train, train_x], axis=1)
    train_y = train.language
    valid_x = pd.concat([scaled_valid, valid_x], axis=1)
    valid_y = validate.language
    rf_train, rf_valid = tune_random_forest(
        train_x, train_y, valid_x, valid_y, max_depth=(2, 31, 1))
    xg_train, xg_valid = tune_gradient_boost(
        train_x, train_y, valid_x, valid_y)
    dt_train, dt_valid = tune_decision_tree(train_x, train_y, valid_x, valid_y)
    fig, axs = plt.subplots(3, 2, sharex='col')
    sns.barplot(data=dt_train, ax=axs[0][0])
    axs[0][0].set_xlabel('max_depth')
    axs[0][0].set_ylabel('Accuracy')
    axs[0, 0].set_title('In Sample Data')
    sns.barplot(data=dt_valid, ax=axs[0][1])
    axs[0][1].set_xlabel('max_depth')
    axs[0][1].set_ylabel('Accuracy')
    axs[0, 1].set_title('Out of Sample Data')
    sns.heatmap(rf_train, ax=axs[1][0])
    axs[1, 0].set_xlabel('max_depth')
    axs[1, 0].set_ylabel('min_samples_leaf')
    axs[1, 0].set_title('In Sample Data')
    sns.heatmap(rf_valid, ax=axs[1, 1])
    axs[1, 1].set_xlabel('max_depth')
    axs[1, 1].set_ylabel('min_samples_leaf')
    axs[1, 1].set_title('Out of Sample Data')
    sns.heatmap(xg_train, ax=axs[2, 0])
    axs[2, 0].set_xlabel('max_depth')
    axs[2, 0].set_ylabel('min_samples_leaf')
    axs[2, 0].set_title('In Sample Data')
    sns.heatmap(xg_valid, ax=axs[2, 1])
    axs[2, 1].set_xlabel('max_depth')
    axs[2, 1].set_ylabel('min_samples_leaf')
    axs[2, 1].set_title('Out of Sample Data')
    plt.show()
