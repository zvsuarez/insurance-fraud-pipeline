
import numpy as np
import pandas as pd
import datetime
from sklearn.base import BaseEstimator, TransformerMixin


class TemporalCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, variable):
        if not isinstance(variable, list):
            raise ValueError('Variables should be a list')
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        pass


class CyclicalTransform(BaseEstimator, TransformerMixin):
    def __init__(self, variable):
        if not isinstance(variable, list):
            raise ValueError('Variables should be a list')
        
    def fit(self, X, y=None):
        pass

    def transform(self, X):
        pass


class Mapper(BaseEstimator, TransformerMixin):
    def __init__(self, variable, mappings):
        if not isinstance(variable, list):
            raise ValueError('Variables should be a list')
        
        if not isinstance(mappings, dict):
            raise ValueError('Mapping should be a dictionary')
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        pass


