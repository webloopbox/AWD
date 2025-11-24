from sklearn.preprocessing import MinMaxScaler
from .i_data_normalizer import IDataNormalizer

class MinMaxNormalizer(IDataNormalizer):
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def fit(self, X):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        return self.scaler.transform(X)

    def fit_transform(self, X):
        return self.scaler.fit_transform(X)
