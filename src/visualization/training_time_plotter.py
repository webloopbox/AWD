import matplotlib.pyplot as plt
import numpy as np
from .i_visualizer import IVisualizer

class TrainingTimePlotter(IVisualizer):
    def __init__(self, figsize=(10, 6)):
        self.figsize = figsize

    def plot(self, time_dict, title='Porównanie Czasu Treningu'):
        fig, ax = plt.subplots(figsize=self.figsize)
        models = list(time_dict.keys())
        times = list(time_dict.values())
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        bars = ax.barh(models, times, color=colors)
        for i, (bar, time) in enumerate(zip(bars, times)):
            ax.text(time, i, f' {time:.3f}s', va='center', fontsize=10)
        ax.set_xlabel('Czas Treningu (sekundy)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        return fig
