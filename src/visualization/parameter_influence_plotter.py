import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .i_visualizer import IVisualizer

class ParameterInfluencePlotter(IVisualizer):
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize

    def plot(self, cv_results, param_name, metric_key=None, title='Wpływ Parametru', ylabel='Średnia wartość (CV)'):
        fig, ax = plt.subplots(figsize=self.figsize)
        param_key = f'param_{param_name}'

        df = self._prepare_dataframe(cv_results)
        if df is None:
            return None

        metric_key = self._get_metric_key(df, metric_key)
        if metric_key is None or param_key not in df.columns:
            return None

        df[param_key] = df[param_key].apply(self._clean_param)
        plot_df, param_codes, code_to_label, is_numeric = self._prepare_plot_df(df, param_key, metric_key)
        if plot_df.empty:
            return None

        means, stds = self._calculate_stats(plot_df, df, param_key, metric_key, is_numeric, param_codes)
        if means.empty:
            return None

        self._plot_data(ax, plot_df, means, stds, metric_key, param_name, ylabel, title, code_to_label)
        plt.tight_layout()
        return fig

    def _prepare_dataframe(self, cv_results):
        if isinstance(cv_results, dict):
            return pd.DataFrame(cv_results)
        if isinstance(cv_results, pd.DataFrame):
            return cv_results.copy()
        return None

    def _get_metric_key(self, df, metric_key):
        if metric_key is None:
            if 'mean_test_accuracy' in df.columns:
                return 'mean_test_accuracy'
            if 'mean_test_score' in df.columns:
                return 'mean_test_score'
        return metric_key

    def _clean_param(self, p):
        if isinstance(p, (np.ma.MaskedArray, np.ma.core.MaskedConstant)):
            return np.nan
        return p

    def _encode_params(self, series):
        numeric = pd.to_numeric(series, errors='coerce')
        if numeric.notna().all():
            numeric = numeric.astype(float)
            unique_vals = np.sort(numeric.unique())
            value_to_code = {float(val): float(val) for val in unique_vals}
            code_to_label = {float(val): f'{val:g}' for val in unique_vals}
            return numeric, value_to_code, code_to_label, True

        string_vals = series.astype(str)
        categories = pd.Categorical(string_vals, ordered=True)
        codes = categories.codes.astype(float)
        value_to_code = {cat: float(idx) for idx, cat in enumerate(categories.categories)}
        code_to_label = {float(idx): cat for idx, cat in enumerate(categories.categories)}
        return codes.astype(float), value_to_code, code_to_label, False

    def _prepare_plot_df(self, df, param_key, metric_key):
        plot_df = df.loc[:, [param_key, metric_key]].copy()
        plot_df[param_key] = plot_df[param_key].apply(self._clean_param)
        plot_df = plot_df.dropna(subset=[param_key, metric_key])
        if plot_df.empty:
            return pd.DataFrame(), None, None, False

        param_codes, value_to_code, code_to_label, is_numeric = self._encode_params(plot_df[param_key])
        plot_df = plot_df.assign(_param_code=param_codes)
        plot_df = plot_df.dropna(subset=['_param_code'])
        return plot_df.sort_values('_param_code'), value_to_code, code_to_label, is_numeric

    def _calculate_stats(self, plot_df, df, param_key, metric_key, is_numeric, value_to_code):
        grouped = plot_df.groupby('_param_code', sort=True)
        means = grouped[metric_key].mean()
        if means.empty:
            return pd.Series(), None

        std_metric_key = metric_key.replace('mean_', 'std_')
        stds = None
        if std_metric_key in df.columns:
            std_plot_df = df.loc[:, [param_key, std_metric_key]].copy()
            std_plot_df[param_key] = std_plot_df[param_key].apply(self._clean_param)
            std_plot_df = std_plot_df.dropna(subset=[param_key, std_metric_key])
            if not std_plot_df.empty:
                if is_numeric:
                    std_plot_df['_param_code'] = pd.to_numeric(std_plot_df[param_key], errors='coerce').map(value_to_code)
                else:
                    std_plot_df['_param_code'] = std_plot_df[param_key].astype(str).map(value_to_code)
                std_plot_df = std_plot_df.dropna(subset=['_param_code', std_metric_key])
                if not std_plot_df.empty:
                    stds = std_plot_df.groupby('_param_code')[std_metric_key].mean()
        return means, stds

    def _plot_data(self, ax, plot_df, means, stds, metric_key, param_name, ylabel, title, code_to_label):
        x_values = means.index.to_numpy(dtype=float)
        ax.scatter(plot_df['_param_code'], plot_df[metric_key], alpha=0.4, s=40, color='#1f77b4', edgecolors='k', linewidths=0.4, label='Wyniki prób')
        ax.plot(x_values, means.values, marker='o', color='#d62728', linewidth=2, label='Średnia')

        if stds is not None and not stds.empty:
            stds = stds.reindex(means.index)
            valid = stds.notna()
            if valid.any():
                lower = means[valid].values - stds[valid].values
                upper = means[valid].values + stds[valid].values
                ax.fill_between(means.index[valid].to_numpy(dtype=float), lower, upper, alpha=0.2, color='#ff9896')

        ax.set_xlabel(param_name, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(frameon=True)
        tick_labels = [code_to_label.get(float(val), str(val)) for val in x_values]
        ax.set_xticks(x_values)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
