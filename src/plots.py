"""
Plotting utilities shared across all experiment scripts.

All plots are saved to the project's plots/ directory. File paths are
constructed from PLOTS_DIR (from config.py) so they work regardless
of where the scripts are launched from.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import FEATURE_DISPLAY_NAMES, PLOTS_DIR


def _ensure_plots_dir() -> None:
    """Make sure the plots/ directory exists before saving figures."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def save_and_show(fig: plt.Figure, filename: str) -> None:
    """Save the figure to plots/<filename> at 150 DPI, then display it.

    filename should be just the file name (e.g. '01_nn_fnr_fpr.png'),
    not a full path. The figure is saved under PLOTS_DIR.
    """
    _ensure_plots_dir()
    out_path: Path = PLOTS_DIR / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


# ───────────────────────────── Classification plots ─────────────────────────────

def plot_heatmap_grid_3d(
    matrices: np.ndarray,
    titles: list[str],
    suptitle: str,
    vmax: float | None = None,
    filename: str | None = None,
) -> None:
    """Render a row of 3 heatmaps side by side.

    Used for neural-network grid search results, where the third axis
    of the matrix indexes the activation function (ReLU / Sigmoid / tanh).

    Parameters
    ----------
    matrices : array of shape (N, N, 3) — one heatmap per activation
    titles   : list of 3 strings — title per subplot
    suptitle : overall figure title
    vmax     : optional upper bound for the colour scale (keeps plots comparable)
    filename : if given, saves to plots/<filename>
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for k in range(3):
        im = axes[k].imshow(
            matrices[:, :, k],
            vmin=0 if vmax is not None else None,
            vmax=vmax,
            aspect="auto",
            origin="lower",
        )
        axes[k].set_title(titles[k])
        axes[k].set_xlabel("N. neurons, Layer 2")
        axes[k].set_ylabel("N. neurons, Layer 1")
        plt.colorbar(im, ax=axes[k])

    plt.suptitle(suptitle, fontsize=14)
    plt.tight_layout()

    if filename:
        save_and_show(fig, filename)
    else:
        plt.show()


def plot_metric_vs_lambda(
    lambdas: np.ndarray,
    values: np.ndarray,
    labels: list[str],
    metric_name: str,
    suptitle: str,
    color: str = "C0",
    filename: str | None = None,
) -> None:
    """Plot a metric against lambda for multiple regularisation types.

    Used for logistic-regression sweeps over lambda values.

    Parameters
    ----------
    lambdas : array of lambda values (log scale on x-axis)
    values  : array of shape (len(lambdas), len(labels))
    labels  : list of subplot titles (one per regularisation type)
    """
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    # If only one subplot, axes isn't an array — normalise
    if n == 1:
        axes = [axes]

    for k in range(n):
        axes[k].semilogx(lambdas, values[:, k], "-o", markersize=4, color=color)
        axes[k].set_title(f"{metric_name} - {labels[k]}")
        axes[k].set_xlabel("Lambda")
        axes[k].set_ylabel(metric_name)
        axes[k].grid(True)

    plt.suptitle(suptitle, fontsize=14)
    plt.tight_layout()

    if filename:
        save_and_show(fig, filename)
    else:
        plt.show()


def plot_svm_heatmap(
    matrix: np.ndarray,
    c_vals: np.ndarray,
    g_vals: np.ndarray,
    title: str,
    filename: str | None = None,
) -> None:
    """Render a single heatmap over a C × gamma grid (for SVM)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    extent = [
        np.log10(g_vals[0]),
        np.log10(g_vals[-1]),
        np.log10(c_vals[0]),
        np.log10(c_vals[-1]),
    ]
    im = ax.imshow(matrix, aspect="auto", origin="lower", extent=extent)
    ax.set_title(title)
    ax.set_xlabel("log10(KernelScale / gamma)")
    ax.set_ylabel("log10(C)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    if filename:
        save_and_show(fig, filename)
    else:
        plt.show()


# ───────────────────────────── Regression plots ─────────────────────────────

def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    filename: str | None = None,
) -> None:
    """Scatter plot of predicted vs actual values with ±25% tolerance bands."""
    maxv = max(y_true.max(), y_pred.max()) * 1.05
    x = np.linspace(0, maxv, 300)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, s=18, marker="x", label="Predictions")
    ax.plot(x, x, "-r", linewidth=2, label="Ideal y=x")
    ax.plot(x, 1.25 * x, "--b", linewidth=1.2, label="+25%")
    ax.plot(x, 0.75 * x, "--b", linewidth=1.2, label="-25%")
    ax.set_xlabel("Actual Grain Size")
    ax.set_ylabel("Predicted Grain Size")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.set_xlim(0, maxv)
    ax.set_ylim(0, maxv)
    ax.grid(True)
    plt.tight_layout()

    if filename:
        save_and_show(fig, filename)
    else:
        plt.show()


def plot_standardised_residuals(
    y_pred: np.ndarray,
    residuals_std: np.ndarray,
    title: str,
    filename: str | None = None,
) -> None:
    """Plot standardised residuals vs predicted values, with ±2 outlier lines."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residuals_std, s=20)
    ax.axhline(2, color="red", linestyle="--", linewidth=1.5)
    ax.axhline(-2, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Predicted Value")
    ax.set_ylabel("Standardised Residuals")
    ax.set_title(title)
    ax.grid(True)
    plt.tight_layout()

    if filename:
        save_and_show(fig, filename)
    else:
        plt.show()


def plot_feature_importance(
    importances: np.ndarray,
    title: str,
    filename: str | None = None,
) -> None:
    """Bar chart of (permutation) feature importance."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(FEATURE_DISPLAY_NAMES, importances, color="steelblue", edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel("Input Variable")
    ax.set_ylabel("Mean Importance")
    plt.xticks(rotation=30)
    plt.tight_layout()

    if filename:
        save_and_show(fig, filename)
    else:
        plt.show()
