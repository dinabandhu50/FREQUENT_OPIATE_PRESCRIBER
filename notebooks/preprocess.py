from sklearn.base import BaseEstimator,TransformerMixin 
import pandas as pd 

class TextPreprocess(BaseEstimator,TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self,X,y=None):
        return self 

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].apply(
                lambda x: ''.join(str(x).split('.'))
            )

        return X
