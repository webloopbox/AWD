import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
)

class HyperparameterTuner:
    def __init__(self, search_type='grid', n_iter=20, cv=5, random_state=42):
        self.search_type = search_type
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state
        self.best_params_ = None
        self.best_score_ = None

    def tune(self, model, X, y, param_space):
        n_classes = len(np.unique(y))

        def _safe_roc_auc(y_true, y_proba, **kwargs):
            try:
                y_proba = np.asarray(y_proba)
                unique_classes = np.unique(y_true)
                if y_proba.ndim == 2 and y_proba.shape[1] > 1:
                    if unique_classes.size == 2:
                        y_proba = y_proba[:, 1]
                    elif unique_classes.size == y_proba.shape[1]:
                        return roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
                    else:
                        return np.nan
                return roc_auc_score(y_true, y_proba)
            except ValueError:
                return np.nan

        def _safe_precision_macro(y_true, y_pred, **kwargs):
            unique_classes = np.unique(y_true)
            if unique_classes.size < n_classes:
                return np.nan
            return precision_score(y_true, y_pred, average='macro', zero_division=0)

        def _safe_recall_macro(y_true, y_pred, **kwargs):
            unique_classes = np.unique(y_true)
            if unique_classes.size < n_classes:
                return np.nan  
            return recall_score(y_true, y_pred, average='macro', zero_division=0)

        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision_macro': make_scorer(_safe_precision_macro),
            'recall_macro': make_scorer(_safe_recall_macro),
            'f1_macro': make_scorer(f1_score, average='macro', zero_division=0),
            'roc_auc_ovr': make_scorer(_safe_roc_auc, needs_proba=True),
            'neg_mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False),
            'neg_mean_absolute_error': make_scorer(mean_absolute_error, greater_is_better=False)
        }

        if self.search_type == 'grid':
            search = GridSearchCV(
                model.model,
                param_space,
                cv=self.cv,
                scoring=scoring,
                refit='accuracy',
                n_jobs=-1,
                verbose=1,
                return_train_score=True
            )
        else:
            search = RandomizedSearchCV(
                model.model,
                param_space,
                n_iter=self.n_iter,
                cv=self.cv,
                scoring=scoring,
                refit='accuracy',
                n_jobs=-1,
                random_state=self.random_state,
                verbose=1,
                return_train_score=True
            )
        search.fit(X, y)
        self.best_params_ = search.best_params_
        self.best_score_ = search.best_score_
        model.set_params(**self.best_params_)
        return model, search.cv_results_
