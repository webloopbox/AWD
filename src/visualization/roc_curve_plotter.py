import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from .i_visualizer import IVisualizer

class ROCCurvePlotter(IVisualizer):
    def __init__(self, figsize=(10, 8)):
        self.figsize = figsize

    def plot(self, y_true, y_pred_proba, n_classes, title='Krzywe ROC', class_names=None):
        fig, ax = plt.subplots(figsize=self.figsize)
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'Krzywa ROC (AUC = {roc_auc:.2f})')
        else:
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                class_label = class_names[i] if class_names and i < len(class_names) else f'Klasa {i}'
                ax.plot(fpr, tpr, label=f'{class_label} (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Losowy')
        ax.set_xlabel('Wskaźnik Fałszywie Pozytywnych', fontsize=12)
        ax.set_ylabel('Wskaźnik Prawdziwie Pozytywnych', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig
