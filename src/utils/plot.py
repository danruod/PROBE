#!/usr/bin/env python3
"""Generation plotting helpers."""

import os
import json
from typing import Optional, Union, List, Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def _step_plot_stats(step: Dict) -> Dict[str, float]:
    """Derive fitness / uncertainty summary stats for one history step.

    Preferred keys from ``gradient_refine``:
      scores / seq_means: per-sequence ensemble mean fitness
      sigma  / seq_uncs:  per-sequence ensemble std (uncertainty)
    Falls back to precomputed ``mean`` / ``mean_std`` / ``uncertainty``.
    """
    if 'seq_means' in step and 'seq_uncs' in step:
        seq_means = np.asarray(step['seq_means'], dtype=np.float64)
        seq_uncs = np.asarray(step['seq_uncs'], dtype=np.float64)
        return {
            'mean': float(seq_means.mean()),
            'mean_std': float(seq_means.std()) if seq_means.size > 1 else 0.0,
            'uncertainty': float(seq_uncs.mean()),
            'uncertainty_std': float(seq_uncs.std()) if seq_uncs.size > 1 else 0.0,
        }

    scores = np.asarray(step.get('scores', []), dtype=np.float64)
    sigma = np.asarray(step.get('sigma', []), dtype=np.float64)

    if scores.size:
        mean = float(np.mean(scores))
        mean_std = float(np.std(scores)) if scores.size > 1 else 0.0
    elif 'mean_std' in step:
        mean = float(step.get('mean', np.nan))
        mean_std = float(step['mean_std'])
    else:
        mean = float(step.get('mean', np.nan))
        mean_std = np.nan

    if sigma.size:
        unc = float(np.mean(sigma))
        unc_std = float(np.std(sigma)) if sigma.size > 1 else 0.0
    elif 'uncertainty' in step:
        unc = float(step['uncertainty'])
        unc_std = float(step.get('uncertainty_std', np.nan))
    else:
        unc = float(step.get('model_sigma', np.nan))
        unc_std = np.nan

    return {
        'mean': mean,
        'mean_std': mean_std,
        'uncertainty': unc,
        'uncertainty_std': unc_std,
    }


def _load_series(
    data_path: Union[str, List[Dict]],
    label: str,
) -> Dict:
    """Load one score-history series into arrays ready for plotting."""
    if isinstance(data_path, list):
        history = data_path
    else:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Score history file not found: {data_path}")
        with open(data_path, 'r') as f:
            history = json.load(f)

    if not history:
        raise ValueError(f"Empty score history: {label}")

    steps = np.asarray([d['step'] for d in history], dtype=np.float64)
    means, mean_stds, uncs, unc_stds = [], [], [], []
    n_families, kmer_div_ref = [], []
    for d in history:
        stats = _step_plot_stats(d)
        means.append(stats['mean'])
        mean_stds.append(stats['mean_std'])
        uncs.append(stats['uncertainty'])
        unc_stds.append(stats['uncertainty_std'])
        n_families.append(d.get('n_families', 1.0))
        kmer_div_ref.append(d.get('kmer_div_ref', 0.0))

    return {
        'label': label,
        'steps': steps,
        'means': np.asarray(means, dtype=np.float64),
        'mean_stds': np.asarray(mean_stds, dtype=np.float64),
        'uncs': np.asarray(uncs, dtype=np.float64),
        'unc_stds': np.asarray(unc_stds, dtype=np.float64),
        'diversity': np.asarray(n_families, dtype=np.float64) * np.asarray(kmer_div_ref, dtype=np.float64),
    }


def _normalize_inputs(
    data_path: Union[str, List[str], List[Dict]],
    show_name: Optional[Union[str, List[str]]],
) -> Tuple[List[Union[str, List[Dict]]], List[str]]:
    """Normalize data_path / show_name into parallel lists."""
    # In-memory history list of step dicts (single series).
    if (
        isinstance(data_path, list)
        and data_path
        and isinstance(data_path[0], dict)
    ):
        paths: List[Union[str, List[Dict]]] = [data_path]
        if show_name is None:
            names = ['run']
        elif isinstance(show_name, str):
            names = [show_name]
        else:
            names = list(show_name)
    elif isinstance(data_path, (list, tuple)):
        paths = list(data_path)
        if show_name is None:
            names = [
                os.path.splitext(os.path.basename(str(p)))[0] for p in paths
            ]
        elif isinstance(show_name, str):
            names = [show_name] * len(paths)
        else:
            names = list(show_name)
    else:
        paths = [data_path]
        if show_name is None:
            names = [os.path.splitext(os.path.basename(str(data_path)))[0]]
        elif isinstance(show_name, str):
            names = [show_name]
        else:
            names = list(show_name)

    if len(names) != len(paths):
        raise ValueError(
            f"show_name length ({len(names)}) must match "
            f"data_path length ({len(paths)})"
        )
    return paths, names


def _style_ax(ax, title: str, ylabel: str, ylabel_color: Optional[str] = None):
    ax.set_title(title, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=11, color=ylabel_color or 'black')
    if ylabel_color is not None:
        ax.tick_params(axis='y', colors=ylabel_color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.15)


def _plot_mean_std(
    ax,
    series_list: List[Dict],
    y_key: str,
    std_key: str,
    colors,
    clip_low: Optional[float] = None,
):
    """Plot mean curves with std bands; one color per series."""
    for series, color in zip(series_list, colors):
        steps = series['steps']
        y = series[y_key]
        std = series[std_key]
        valid = np.isfinite(y) & np.isfinite(std)
        if valid.any():
            lo = y[valid] - std[valid]
            hi = y[valid] + std[valid]
            if clip_low is not None:
                lo = np.clip(lo, clip_low, None)
            # Band first (no legend entry) so lines stay visually dominant.
            ax.fill_between(
                steps[valid], lo, hi,
                alpha=0.18, color=color, linewidth=0, zorder=1,
            )
        sns.lineplot(
            x=steps, y=y, ax=ax,
            color=color, label=series['label'],
            linewidth=2.2, zorder=2,
        )


def _plot_scalar(
    ax,
    series_list: List[Dict],
    y_key: str,
    colors,
):
    """Plot scalar-per-step curves (no std band)."""
    for series, color in zip(series_list, colors):
        steps = series['steps']
        y = series[y_key]
        if not np.isfinite(y).any():
            continue
        sns.lineplot(
            x=steps, y=y, ax=ax,
            color=color, label=series['label'],
            linewidth=2.0, zorder=2,
        )


def plot_score_history(
    data_path: Union[str, List[str], List[Dict]],
    save_path: Optional[str] = None,
    show_name: Optional[Union[str, List[str]]] = None,
) -> str:
    """Plot fitness / uncertainty / diversity from score-history JSON(s).

    Args:
        data_path: path to ``*_score_history.json``, a list of such paths
            (one curve each), or an in-memory history list.
        save_path: output PNG path (required when plotting multiple files
            or an in-memory list).
        show_name: legend label(s) matching ``data_path``. Defaults to file
            stem(s) when omitted.
    """
    paths, names = _normalize_inputs(data_path, show_name)
    series_list = [
        _load_series(path, name) for path, name in zip(paths, names)
    ]

    if save_path is None:
        if len(paths) != 1 or not isinstance(paths[0], str):
            raise ValueError(
                "save_path is required when plotting multiple files "
                "or an in-memory history list"
            )
        save_path = str(paths[0]).replace('.json', '.png')

    panels = ['fitness', 'uncertainty', 'diversity']
    fig, axes = plt.subplots(
        len(panels), 1,
        figsize=(10, 6),
        height_ratios=[2, 1, 1],
        sharex=True,
        gridspec_kw={'hspace': 0.18},
    )
    if len(panels) == 1:
        axes = [axes]
    ax_map = dict(zip(panels, axes))

    # Distinct, colorblind-safe colors; same hue for mean line + std band.
    colors = sns.color_palette('colorblind', n_colors=len(series_list))
    colors = colors[:len(series_list)]
    sns.set_theme(style='ticks', context='notebook')

    # Panel: predicted fitness mean ± std across sequences
    ax_fit = ax_map['fitness']
    _plot_mean_std(ax_fit, series_list, 'means', 'mean_stds', colors)
    _style_ax(ax_fit, None, 'Predicted Fitness')
    ax_fit.legend(frameon=False, title=None)

    # Panel: uncertainty mean ± std across sequences
    ax_unc = ax_map['uncertainty']
    _plot_mean_std(
        ax_unc, series_list, 'uncs', 'unc_stds', colors, clip_low=0.0,
    )
    _style_ax(ax_unc, None, 'Uncertainty')
    ax_unc.set_ylim(bottom=0)
    ax_unc.legend().set_visible(False)

    # Panel: diversity
    ax_div = ax_map['diversity']
    _plot_scalar(ax_div, series_list, 'diversity', colors)
    _style_ax(ax_div, None, 'Diversity')
    ax_div.set_ylim(bottom=0)
    ax_div.legend().set_visible(False)

    axes[-1].set_xlabel('Gradient Step', fontsize=11)

    fig.subplots_adjust(right=0.78, hspace=0.28)
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Score history plot saved: {save_path}")
    return save_path
