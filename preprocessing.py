
import numpy as np
import pandas as pd
import datetime
from sklearn.base import BaseEstimator, TransformerMixin



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


class TemporalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, num_temp, cat_temp):

        if not isinstance(num_temp, list):
            raise ValueError('Variables should be a list')
        
        if not isinstance(cat_temp, list):
            raise ValueError('Variables should be a list')
        
        for item in num_temp:
            if not isinstance(item, (int, float)):
                raise ValueError('Items should be integers or floats')
            
        for item in cat_temp:
            if not isinstance(item, str):
                raise ValueError('Items should be strings')
            
        self.num_temp = num_temp
        self.cat_temp = cat_temp
            
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        pass

