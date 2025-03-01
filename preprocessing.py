
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats

# custom class for cleaning temporal features with mismatched types
class TemporalClean(BaseEstimator, TransformerMixin):
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
            if feature == 'WeekOfMonthClaimed' or feature == 'WeekOfMonth':
                X[feature] = X[feature].replace('0', 1)
            if feature == 'MonthClaimed' or feature == 'Month':
                X[feature] = X[feature].replace('0', 'Jan')

        return X


# custom class for implementing 
# sine/cosine transform to capture cyclical nature of temporal features
class TemporalCycleTransform(BaseEstimator, TransformerMixin):
    def __init__(self, variable):
        if not isinstance(variable, list):
            raise ValueError('Variables should be a list')
        
        self.variable = variable
        
    def fit(self, X, y=None):
        pass

    def transform(self, X):
        X = X.copy()

        for feature in self.variable:
            # create new colummns for sine/cosine transformed values
            if feature == 'MonthClaimed' or feature == 'Month':
                X[feature+'_sin'] = np.sin(2 * np.pi * X[feature] / 12)
                X[feature+'_cos'] = np.cos(2 * np.pi * X[feature] / 12)
            if feature == 'DayOfWeekClaimed' or feature == 'DayOfWeek':
                X[feature+'_sin'] = np.sin(2 * np.pi * X[feature] / 7)
                X[feature+'_cos'] = np.cos(2 * np.pi * X[feature] / 7)

        return X


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


# custom class for transforming the Age column
class AgeTransform(BaseEstimator, TransformerMixin):
    def __init__(self, variable):
        if not isinstance(variable, list):
            raise ValueError('Variables should be a list')
        
        self.variable = variable
        self.bins = [0, 18, 24, 34, 49, 64, 100]
        self.labels = ['Under 18', 'Very young', 'Young adult', 'Middle-aged', 'Older adult', 'Senior']
        
    # calculate the mean of Age excluding the 0 values
    def fit(self, X, y=None):
        self.mean_exc_zero_val = X[X[self.variable] > 0][self.variable].mean().to_dict()
        #                        df[df[['Age', 'Year']]>0][['Age', 'Year']].mean().to_dict()
        return self
    
    def transform(self, X):
        X = X.copy()

        # replace 0 values with the Age mean and apply box-cox transform
        for feature in self.variable:
            #X[feature] = X[feature].replace(0, self.mean_exc_zero_val[feature], inplace=True)
            X[feature] = X[feature].apply(lambda z: self.mean_exc_zero_val[feature] if z <=0 else z)
            X[feature], _ = stats.boxcox(X[feature])
            # create a new feature with discretised bins
            X[feature+'Group'] = pd.cut(X[feature], bins=self.bins, labels=self.labels)
            
        return X
