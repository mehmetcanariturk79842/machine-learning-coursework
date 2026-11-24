import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin


class PercentileClipper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def __init__(self, lower=1, upper=99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.percentile(values, self.lower, axis=0)
        self.upper_bounds_ = np.percentile(values, self.upper, axis=0)
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)