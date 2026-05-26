
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class Missing_data_emulation(BaseEstimator,TransformerMixin):
    def __init__(self,missing_rate=0.15, random_state=42):
        super().__init__()
        self.missing_rate = missing_rate
        self.random_state = random_state

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if not getattr(self, 'dropout_enabled', True):
            return X
        
        X_corrupted = X.copy()
        rng = np.random.default_rng(self.random_state)

        is_df = isinstance(X_corrupted, pd.DataFrame)
        values = X_corrupted.values if is_df else X_corrupted   
        mask = rng.random(size=values.shape) < self.missing_rate
        if is_df:
            for col_idx, col_name in enumerate(X_corrupted.columns):

                X_corrupted.iloc[mask[:, col_idx], col_idx] = np.nan
        else:
            values[mask] = np.nan
            
        return X_corrupted