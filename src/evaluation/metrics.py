import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error,
    roc_auc_score, confusion_matrix, classification_report
)

class MetricsCalculator:
    def __init__(self, average='macro'):
        self.average = average

    def calculate_classification_metrics(self, y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=self.average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=self.average, zero_division=0),
        }

    def calculate_error_metrics(self, y_true, y_pred):
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
        }

    def calculate_roc_auc(self, y_true, y_pred_proba, multi_class='ovr'):
        try:
            if len(np.unique(y_true)) == 2:
                if y_pred_proba.ndim == 2:
                    y_pred_proba = y_pred_proba[:, 1]
                return roc_auc_score(y_true, y_pred_proba)
            else:
                return roc_auc_score(y_true, y_pred_proba, multi_class=multi_class, average=self.average)
        except Exception as e:
            return None

    def get_confusion_matrix(self, y_true, y_pred):
        return confusion_matrix(y_true, y_pred)

    def get_classification_report(self, y_true, y_pred):
        return classification_report(y_true, y_pred)

    def calculate_all_metrics(self, y_true, y_pred, y_pred_proba=None):
        metrics = self.calculate_classification_metrics(y_true, y_pred)
        metrics.update(self.calculate_error_metrics(y_true, y_pred))
        if y_pred_proba is not None:
            roc_auc = self.calculate_roc_auc(y_true, y_pred_proba)
            if roc_auc is not None:
                metrics['roc_auc'] = roc_auc
        metrics['confusion_matrix'] = self.get_confusion_matrix(y_true, y_pred)
        return metrics
