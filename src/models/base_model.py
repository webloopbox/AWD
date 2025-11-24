from abc import ABC, abstractmethod
import time
from .i_model import IModel

class ModelBase(IModel):
    def __init__(self):
        self.model = None
        self.training_time = 0.0
        self.is_trained = False

    def fit(self, X_train, y_train):
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time
        self.is_trained = True
        return self

    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)

    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)

    def get_params(self):
        return self.model.get_params()

    def set_params(self, **params):
        self.model.set_params(**params)
        return self

    def get_training_time(self):
        return self.training_time
