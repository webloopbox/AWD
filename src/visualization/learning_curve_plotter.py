import matplotlib.pyplot as plt
import numpy as np
from .i_visualizer import IVisualizer

class LearningCurvePlotter(IVisualizer):
    def __init__(self, figsize=(10, 6)):
        self.figsize = figsize

    def plot(self, train_sizes, train_scores, test_scores, title='Krzywe Uczenia'):
        fig, ax = plt.subplots(figsize=self.figsize)
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        ax.plot(train_sizes, train_mean, label='Wynik treningowy', marker='o')
        ax.fill_between(train_sizes, train_mean - train_std,
                       train_mean + train_std, alpha=0.1)
        ax.plot(train_sizes, test_mean, label='Wynik walidacyjny', marker='s')
        ax.fill_between(train_sizes, test_mean - test_std,
                       test_mean + test_std, alpha=0.1)
        ax.set_xlabel('Rozmiar Zbioru Treningowego', fontsize=12)
        ax.set_ylabel('Wynik', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig
