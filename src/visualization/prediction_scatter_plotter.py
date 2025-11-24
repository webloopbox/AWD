import matplotlib.pyplot as plt
import numpy as np
from .i_visualizer import IVisualizer

class PredictionScatterPlotter(IVisualizer):
    def __init__(self, figsize=(10, 8)):
        self.figsize = figsize

    def plot(self, y_true, y_pred, title='Rzeczywiste vs Przewidywane', class_labels=None, jitter=0.15):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        fig, ax = plt.subplots(figsize=self.figsize)

        unique_vals = np.unique(np.concatenate((y_true, y_pred)))
        is_discrete = unique_vals.size <= 20

        if is_discrete:
            self._plot_discrete(ax, y_true, y_pred, class_labels, jitter, unique_vals)
        else:
            self._plot_continuous(ax, y_true, y_pred)

        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def _plot_discrete(self, ax, y_true, y_pred, class_labels, jitter, unique_vals):
        label_order = sorted(unique_vals.tolist())
        label_to_idx = {val: idx for idx, val in enumerate(label_order)}
        rng = np.random.default_rng(42)
        x_idx = np.array([label_to_idx[val] for val in y_true])
        y_idx = np.array([label_to_idx[val] for val in y_pred])
        x_pos = x_idx + rng.uniform(-jitter, jitter, size=x_idx.shape)
        y_pos = y_idx + rng.uniform(-jitter, jitter, size=y_idx.shape)
        correct_mask = y_true == y_pred

        self._scatter_discrete(ax, x_pos, y_pos, correct_mask)
        self._set_discrete_labels_and_ticks(ax, label_order, class_labels)
        ax.set_xlabel('Klasa rzeczywista', fontsize=12)
        ax.set_ylabel('Klasa przewidywana', fontsize=12)
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.7)

    def _scatter_discrete(self, ax, x_pos, y_pos, correct_mask):
        handles, labels = [], []
        if np.any(correct_mask):
            correct_scatter = ax.scatter(x_pos[correct_mask], y_pos[correct_mask], color='#2ca02c', edgecolors='k', alpha=0.7, s=60)
            handles.append(correct_scatter)
            labels.append('Poprawne')
        if np.any(~correct_mask):
            incorrect_scatter = ax.scatter(x_pos[~correct_mask], y_pos[~correct_mask], color='#d62728', edgecolors='k', alpha=0.8, s=60)
            handles.append(incorrect_scatter)
            labels.append('Błędne')
        if handles:
            ax.legend(handles, labels, loc='upper left', frameon=True)

    def _format_labels(self, values, class_labels):
        if isinstance(class_labels, dict):
            return [class_labels.get(v, str(v)) for v in values]
        if isinstance(class_labels, (list, tuple)):
            label_map = {idx: lbl for idx, lbl in enumerate(class_labels)}
            return [label_map.get(int(v), str(v)) for v in values]
        return [str(v) for v in values]

    def _set_discrete_labels_and_ticks(self, ax, label_order, class_labels):
        tick_labels = self._format_labels(label_order, class_labels)
        tick_positions = range(len(label_order))
        ax.set_xticks(list(tick_positions))
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        ax.set_yticks(list(tick_positions))
        ax.set_yticklabels(tick_labels)
        ax.set_xlim(-0.6, len(label_order) - 0.4)
        ax.set_ylim(-0.6, len(label_order) - 0.4)

    def _plot_continuous(self, ax, y_true, y_pred):
        scatter = ax.scatter(y_true, y_pred, alpha=0.6, c=np.abs(y_true - y_pred), cmap='coolwarm', edgecolors='k', s=50)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfekcyjna predykcja')
        ax.set_xlabel('Wartości rzeczywiste', fontsize=12)
        ax.set_ylabel('Wartości przewidywane', fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Błąd bezwzględny')
