import matplotlib.pyplot as plt
import seaborn as sns
from .i_visualizer import IVisualizer

class ConfusionMatrixPlotter(IVisualizer):
    def __init__(self, figsize=(8, 6)):
        self.figsize = figsize

    def plot(self, cm, class_names=None, title='Macierz Pomyłek'):
        fig, ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names if class_names else 'auto',
                   yticklabels=class_names if class_names else 'auto',
                   ax=ax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Prawdziwa Etykieta', fontsize=12)
        ax.set_xlabel('Przewidywana Etykieta', fontsize=12)
        plt.tight_layout()
        return fig
