from sklearn.preprocessing import LabelEncoder as SklearnLabelEncoder

class LabelEncoder:
    def __init__(self):
        self.encoder = SklearnLabelEncoder()
        self.classes_ = None

    def fit(self, y):
        self.encoder.fit(y)
        self.classes_ = self.encoder.classes_
        return self

    def transform(self, y):
        return self.encoder.transform(y)

    def fit_transform(self, y):
        encoded = self.encoder.fit_transform(y)
        self.classes_ = self.encoder.classes_
        return encoded

    def inverse_transform(self, y):
        return self.encoder.inverse_transform(y)
