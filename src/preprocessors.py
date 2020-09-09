from sklearn.base import BaseEstimator,TransformerMixin 
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder

import pandas as pd 
import numpy as np

# categorical encoders
class RareLabelCatEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, tol=0.05, cols=None):
        
        self.tol = tol
        
        if not isinstance(cols, list):
            self.variables = [cols]
        else:
            self.variables = cols

    def fit(self, X, y=None):

        # persist frequent labels in dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            # the encoder will learn the most frequent categories
            t = pd.Series(X[var].value_counts() / np.float(len(X)))
            # frequent labels:
            self.encoder_dict_[var] = list(t[t >= self.tol].index)

        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(X[feature].isin(self.encoder_dict_[
                    feature]), X[feature], 'Rare')

        return X

class FrequencyCatEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, cols=None):        
        if not isinstance(cols, list):
            self.variables = [cols]
        else:
            self.variables = cols

    def fit(self, X, y=None):

        # persist frequent labels in dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            self.encoder_dict_[var] = X[var].value_counts().to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = X[var].apply(lambda x: self.encoder_dict_[var][x])

        return X

    def fit_transform(self,X,y=None):
        self.fit(X)
        X = self.transform(X)
        return X


class OneHotCatEncoder(BaseEstimator,TransformerMixin):

    def __init__(self,cols=None,**kwargs):
        if not isinstance(cols,list):
            self.variables = [cols]
        else:
            self.variables = cols

        self.encoder = {}
        for var in self.variables:
            self.encoder[var] = OneHotEncoder(sparse=False,**kwargs)
    
    def fit(self,X,y=None):
        for var in self.variables:
            self.encoder[var].fit(X[var].values.reshape(-1,1))
        return self
    
    def transform(self, X):
        X = X.copy().reset_index(drop=True)
        for var in self.variables:
            X_ohe = self.encoder[var].transform(X[var].values.reshape(-1,1))
            X = pd.concat([X.drop(var,axis=1),pd.DataFrame(X_ohe).add_prefix(var+'_')],axis=1)
        return X

    def fit_transform(self,X,y=None):
        self.fit(X)
        X = self.transform(X)
        return X


# Dimension reduction
class PCATransformer(BaseEstimator,TransformerMixin):

    def __init__(self,cols=None,**kwargs):
        if not isinstance(cols,list):
            self.variables = [cols]
        else:
            self.variables = cols
        self.encoder = PCA(**kwargs)
    
    def fit(self,X,y=None):
        self.encoder.fit(X[self.variables])
        return self
    
    def transform(self, X):
        X = X.copy()
        X_pca = self.encoder.transform(X[self.variables])

        # making dataframe, replace old feature with pca feature
        X_transformed = pd.concat([X.drop(self.variables,axis=1),pd.DataFrame(X_pca).add_prefix('pca_')],axis=1)
        return X_transformed

    def fit_transform(self,X,y=None):
        self.fit(X)
        X = self.transform(X)
        return X


# utilities
class FeaturesToDrop(BaseEstimator, TransformerMixin):

    def __init__(self, cols=None):
        
        self.variables = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        # if self.variables in list(X.columns):
        X = X.drop(self.variables, axis=1)
        return X 

    def fit_transform(self,X,y=None):
        self.fit(X)
        X = self.transform(X)
        return X
    

class FeaturesToKeep(BaseEstimator, TransformerMixin):

    def __init__(self, cols=None):
        
        self.variables = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        X = X[self.variables]
        return X

    def fit_transform(self,X,y=None):
        self.fit(X)
        X = self.transform(X)
        return X



# if needed

# class DummyCatEncoder(BaseEstimator,TransformerMixin):

#     def __init__(self,cols=None,**kwargs):
#         if not isinstance(cols,list):
#             self.variables = [cols]
#         else:
#             self.variables = cols
#         self.kwargs = kwargs
    
#     def fit(self,X,y=None):
#         return self
    
#     def transform(self, X):
#         X = X.copy()
#         X = pd.get_dummies(X,columns=self.variables,**self.kwargs)
#         return X

#     def fit_transform(self,X,y=None):
#         self.fit(X)
#         X = self.transform(X)
#         return X