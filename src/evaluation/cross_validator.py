import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from .i_validator import IValidator

class CrossValidator(IValidator):
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def validate(self, model, X, y):
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
            'recall_macro': make_scorer(recall_score, average='macro', zero_division=0),
            'f1_macro': make_scorer(f1_score, average='macro', zero_division=0)
        }
        scores = cross_validate(
            model.model,
            X,
            y,
            cv=self.kfold,
            scoring=scoring,
            return_train_score=True
        )
        return self._compute_statistics(scores)

    def _compute_statistics(self, scores):
        results = {}
        for metric in scores.keys():
            if metric.startswith('test_'):
                metric_name = metric.replace('test_', '')
                results[f'{metric_name}_mean'] = np.mean(scores[metric])
                results[f'{metric_name}_std'] = np.std(scores[metric])
        return results
