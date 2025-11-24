import numpy as np
from sklearn.model_selection import train_test_split

class DataSplitter:
    def __init__(self, test_size=0.1, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, y):
        return train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if len(np.unique(y)) > 1 else None
        )
