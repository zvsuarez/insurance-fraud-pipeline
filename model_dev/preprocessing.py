
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# custom class for cleaning temporal features with mismatched types
class CleanTransform(BaseEstimator, TransformerMixin):
    def __init__(self, variable):
        # check if the features are in a list
        if not isinstance(variable, list):
            raise ValueError('Variables should be a list')
        
        self.variable = variable
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # replace features that should not have 0 values
        for feature in self.variable:
            if feature == 'DayOfWeekClaimed' or feature == 'DayOfWeek':
                X[feature] = X[feature].replace('0', 'Monday')
            if feature == 'MonthClaimed' or feature == 'Month':
                X[feature] = X[feature].replace('0', 'Jan')

        return X


# custom class for implementing 
# sine/cosine transform to capture cyclical nature of temporal features
class CoSineTransform(BaseEstimator, TransformerMixin):
    def __init__(self, variable):
        if not isinstance(variable, list):
            raise ValueError('Variables should be a list')
        
        self.variable = variable
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for feature in self.variable:
            # create new colummns for sine/cosine transformed values
            if feature == 'DayOfWeekClaimed' or feature == 'DayOfWeek':
                X[feature+'_sin'] = np.sin(2 * np.pi * X[feature] / 7)
                X[feature+'_cos'] = np.cos(2 * np.pi * X[feature] / 7)
            if feature == 'MonthClaimed' or feature == 'Month':
                X[feature+'_sin'] = np.sin(2 * np.pi * X[feature] / 12)
                X[feature+'_cos'] = np.cos(2 * np.pi * X[feature] / 12)
            
        return X


# custom class for dropping columns
class DropTransform(BaseEstimator, TransformerMixin):
    def __init__(self, variable):
        if not isinstance(variable, list):
            raise ValueError('Variables should be a list')
        
        self.variable = variable
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # inplace=False is default so automatically returns a copy
        return X.drop(columns=self.variable)


# custom class for features with mappings
class MapTransform(BaseEstimator, TransformerMixin):
    def __init__(self, variable, mappings):
        if not isinstance(variable, list):
            raise ValueError('Variables should be a list')
        
        if not isinstance(mappings, dict):
            raise ValueError('Mapping should be a dictionary')
        
        self.variable = variable
        self.mappings = mappings
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()

        for feature in self.variable:
            X[feature] = X[feature].map(self.mappings)
        
        return X
    
    def get_feature_names_out(self, input_features=None):
        # Return a list with the name of the output features
        return self.variable
    

"""class PandasWrapper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return pd.DataFrame(X, columns=X.columns)"""
