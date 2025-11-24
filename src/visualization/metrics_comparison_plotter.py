import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .i_visualizer import IVisualizer

class MetricsComparisonPlotter(IVisualizer):
    def __init__(self, figsize=(12, 6)):
        self.figsize = figsize

    def plot(self, metrics_dict, title='Porównanie Metryk Modeli'):
        fig, ax = plt.subplots(figsize=self.figsize)
        df = pd.DataFrame(metrics_dict).T
        x = np.arange(len(df.columns))
        width = 0.35
        for i, model in enumerate(df.index):
            offset = width * (i - len(df.index) / 2 + 0.5)
            ax.bar(x + offset, df.loc[model], width, label=model)
        ax.set_xlabel('Metryki', fontsize=12)
        ax.set_ylabel('Wynik', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df.columns, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        return fig
