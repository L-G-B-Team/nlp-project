import numpy as np
import pandas as pd
from typing import Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import NotFittedError

ModelType = Union[LogisticRegression, DecisionTreeClassifier,
                  RandomForestClassifier, KNeighborsClassifier]

def model_data(model:ModelType, features: pd.DataFrame,target:Union[pd.Series,None]= None,result_suffix:str = '')->pd.DataFrame:
    #TODO Docstring
    y_hat = pd.Series()
    try:
        y_hat = pd.Series(model.predict(features))
    except NotFittedError:
        if target is None:
            raise NotFittedError('Model not fit and target not provided')
        model.fit(features,target)
        y_hat = pd.Series(model.predict(features))
    y_hat.name = 'yhat' + ('_' + result_suffix if len(result_suffix) > 0 else '' )
    y_hat.index = target.index
    return pd.concat([target,y_hat],axis=1)
