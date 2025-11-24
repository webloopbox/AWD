import numpy as np
import pandas as pd
from .i_data_normalizer import IDataNormalizer
from .label_encoder import LabelEncoder

class DataPreprocessor:
    def __init__(self, normalizer: IDataNormalizer, label_encoder: LabelEncoder):
        self.normalizer = normalizer
        self.label_encoder = label_encoder
        self.feature_names = None

    def preprocess_features(self, X_train, X_test):
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values
            X_test = X_test.values
        X_train_norm = self.normalizer.fit_transform(X_train)
        X_test_norm = self.normalizer.transform(X_test)
        return X_train_norm, X_test_norm

    def preprocess_labels(self, y_train, y_test):
        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_test_enc = self.label_encoder.transform(y_test)
        return y_train_enc, y_test_enc

    def get_num_classes(self):
        return len(self.label_encoder.classes_)
