from sklearn.preprocessing import StandardScaler
from .i_data_normalizer import IDataNormalizer

class StandardNormalizer(IDataNormalizer):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        return self.scaler.transform(X)

    def fit_transform(self, X):
        return self.scaler.fit_transform(X)
