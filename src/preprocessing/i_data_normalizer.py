from abc import ABC, abstractmethod

class IDataNormalizer(ABC):
    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def transform(self, X):
        pass

    @abstractmethod
    def fit_transform(self, X):
        pass
