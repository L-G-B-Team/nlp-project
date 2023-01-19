import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from IPython.display import Markdown as md
from sklearn.metrics import ConfusionMatrixDisplay
from prepare import prepare_readme
ModelType = Union[LogisticRegression, DecisionTreeClassifier,
                  RandomForestClassifier,
                  GradientBoostingClassifier]
DT_MAX_DEPTH = 22
RF_MAX_DEPTH = 23
RF_MIN_SAMPLES_LEAF = 2
XG_MIN_SAMPLES_LEAF = 9
XG_MAX_DEPTH = 9

def model_data(model: ModelType, features: pd.DataFrame, target: Union[pd.Series, None] = None, result_suffix: str = '') -> pd.DataFrame:
    '''
    Fits (if applicable) and runs predictions on given model.
    ## Parameters
    model: a `DecisionTreeClassifier`, `RandomForestClassifier`,
    or `GradientBoostingClassifier` to be modeled on (note)
    ## Returns
    
    '''
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
    y_hat.index = features.index 
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
                                    result_suffix=str(leaf))
            valid_yhat = model_data(model, valid_features, valid_target,
                                    result_suffix=str(leaf))
            train_sub_ser[train_yhat.name] = metric(train_target.to_numpy(),
                                                    train_yhat.to_numpy())
            valid_sub_ser[valid_yhat.name] = metric(valid_target.to_numpy(),
                                                    valid_yhat.to_numpy())
        train_ser['n_estimators_' + str(depth)] = train_sub_ser
        valid_ser['n_estimators_' + str(depth)] = valid_sub_ser
    return (pd.DataFrame(train_ser),
            pd.DataFrame(valid_ser))


def scale(features: pd.Series, scaler: MinMaxScaler) -> pd.Series:
    '''
    Fits (if applicable), and scales data with
    ## Parameters
    
    ## Returns
    
    '''
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
    ret_train = encode_has_language(train)
    ret_validate = encode_has_language(validate)
    ret_test = encode_has_language(test)
    return train, validate, test


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


def get_features_and_target(train: pd.DataFrame, validate: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    #TODO DOCSTRING
    # encode_has_language
    training_x = encode_has_language(train)
    validate_x = encode_has_language(validate)
    testing_x = encode_has_language(test)
    # scale lemmatized_len
    scaler = MinMaxScaler()
    scaled_valid = scale(validate.lemmatized_len, scaler)
    scaled_train = scale(train.lemmatized_len, scaler)
    scaled_test = scale(test.lemmatized_len, scaler)
    #get TFIDF
    tfidf = TfidfVectorizer(ngram_range=(1,2))
    train_tfidf = tf_idf(train.lemmatized,tfidf)
    valid_tfidf = tf_idf(validate.lemmatized,tfidf)
    test_tfidf = tf_idf(test.lemmatized,tfidf)
    # concat scaled_train,tfidf, and encoded_has_language
    train_x = pd.concat([train_tfidf,scaled_train],axis=1)
    train_x = pd.concat([train_x,training_x],axis=1)
    train_y = train.language
    valid_x = pd.concat([valid_tfidf,scaled_valid, validate_x], axis=1)
    valid_y = validate.language
    test_x = pd.concat([test_tfidf,scaled_test, testing_x], axis=1)
    test_y = test.language
    return train_x, train_y, valid_x, valid_y,test_x,test_y


def tune_hypers(train: pd.DataFrame, validate: pd.DataFrame) -> plt.Axes:
    # TODO doctring
    # get X and y values:
    train_x, train_y, valid_x, valid_y,_,_ = get_features_and_target(
        train, validate,train)
    return_dct = {}
    return_dct['rf_train'], return_dct['rf_valid'] = tune_random_forest(
        train_x, train_y, valid_x, valid_y, max_depth=(2, 31, 1))
    return_dct['xg_train'], return_dct['xg_valid'] = tune_gradient_boost(
        train_x, train_y, valid_x, valid_y)
    return_dct['dt_train'], return_dct['dt_valid'] = tune_decision_tree(
        train_x, train_y, valid_x, valid_y)
    return return_dct


def model_and_evaluate(features: pd.DataFrame, target: pd.Series, model: ModelType) -> pd.DataFrame:
    yhat = model_data(model, features, target)
    return accuracy_score(target, yhat)

def create_models()->Tuple[GradientBoostingClassifier,RandomForestClassifier,DecisionTreeClassifier]:
    xg_boost = GradientBoostingClassifier(min_samples_leaf=XG_MIN_SAMPLES_LEAF,max_depth=XG_MAX_DEPTH,random_state=27)
    random_forest = RandomForestClassifier(n_estimators=180,min_samples_leaf=RF_MIN_SAMPLES_LEAF,max_depth=RF_MAX_DEPTH,random_state=27)
    decision_tree = DecisionTreeClassifier(max_depth=DT_MAX_DEPTH,random_state=27)
    return xg_boost, random_forest, decision_tree


def compare_models(train_x: pd.DataFrame, train_y: pd.Series, valid_x: pd.DataFrame, valid_y: pd.Series, decision_tree: DecisionTreeClassifier,
                   random_forest: RandomForestClassifier, xg_boost: GradientBoostingClassifier) -> pd.DataFrame:
    # TODO Docstring
    ret_dict = {}
    sub_dct = {}
    sub_dct['Train'] = model_and_evaluate(train_x, train_y, decision_tree)
    sub_dct['Validate'] = model_and_evaluate(valid_x, valid_y, decision_tree)
    ret_dict['Decision Tree'] = sub_dct.copy()
    sub_dct['Train'] = model_and_evaluate(train_x, train_y, random_forest)
    sub_dct['Validate'] = model_and_evaluate(valid_x, valid_y, random_forest)
    ret_dict['Random Forest'] = sub_dct.copy()
    sub_dct['Train'] = model_and_evaluate(train_x, train_y, xg_boost)
    sub_dct['Validate'] = model_and_evaluate(valid_x, valid_y, xg_boost)
    ret_dict['Gradient Boosting'] = sub_dct
    return pd.DataFrame(ret_dict)


def plot_data(data: Dict[str, pd.DataFrame]) -> None:
    # TODO Docstring
    vmax = 1.0
    vmin = .3
    fig, axs = plt.subplots(3, 2, figsize=(10, 30), sharex='col')
    axs[0, 0].bar(data['dt_train'].index, data['dt_train'].values)
    axs[0][0].set_xlabel('max_depth')
    axs[0][0].set_ylabel('Accuracy')
    axs[0, 0].set_title('In Sample Data')
    axs[0, 1].bar(data['dt_valid'].index, data['dt_valid'].values)
    axs[0][1].set_xlabel('max_depth')
    axs[0][1].set_ylabel('Accuracy')
    axs[0, 1].set_title('Out of Sample Data')
    sns.heatmap(data['rf_train'], ax=axs[1][0], vmin=vmin, vmax=vmax)
    axs[1, 0].set_xlabel('max_depth')
    axs[1, 0].set_ylabel('min_samples_leaf')
    axs[1, 0].set_title('In Sample Data')
    sns.heatmap(data['rf_valid'], ax=axs[1, 1], vmin=vmin, vmax=vmax)
    axs[1, 1].set_xlabel('max_depth')
    axs[1, 1].set_ylabel('min_samples_leaf')
    axs[1, 1].set_title('Out of Sample Data')
    sns.heatmap(data['xg_train'], ax=axs[2, 0], vmin=vmin, vmax=vmax)
    axs[2, 0].set_xlabel('max_depth')
    axs[2, 0].set_ylabel('min_samples_leaf')
    axs[2, 0].set_title('In Sample Data')
    sns.heatmap(data['xg_valid'], ax=axs[2, 1], vmin=vmin, vmax=vmax)
    axs[2, 1].set_xlabel('max_depth')
    axs[2, 1].set_ylabel('min_samples_leaf')
    axs[2, 1].set_title('Out of Sample Data')
    plt.tight_layout()
    plt.show()

def run_test(test_x:pd.DataFrame,test_y:pd.Series,model:ModelType)->ConfusionMatrixDisplay:
    # TODO Docstring
    yhat_test = model_data(model,test_x)
    acc_score = accuracy_score(test_y,yhat_test) * 100
    return md(f'## Accuracy Score: {acc_score:1.2f}%')

def tf_idf(documents:pd.Series,tfidf:TfidfVectorizer)->pd.DataFrame:
    # TODO Docstring
    tfidf_docs = np.empty((0,5))
    try:
        tfidf_docs = tfidf.transform(documents.values)
    except NotFittedError:
        tfidf_docs = tfidf.fit_transform(documents.values)
    return pd.DataFrame(tfidf_docs.todense(),index=documents.index,columns=tfidf.get_feature_names_out())
    
def predict_readme(readme:str,scaler:MinMaxScaler,tfidf:TfidfVectorizer,model:ModelType)->str:
    prepped_readme = prepare_readme(readme)
    readme_ser = pd.DataFrame({'repo':'','lemmatized':prepped_readme},index=[0])
    readme_ser['lemmatized_len'] = scaler.transform(np.array([readme_ser.lemmatized.str.len()]).reshape(1,-1))
    encoded = encode_has_language(readme_ser)
    tfidf = tf_idf(readme_ser.lemmatized,tfidf)
    readme_ser = pd.concat([readme_ser.lemmatized_len,encoded,tfidf],axis=1)
    return model.predict(readme_ser)[0]
