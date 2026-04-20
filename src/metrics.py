"""
Evaluation metrics for both classification and regression tasks.

Classification metrics (for PCG detection):
  - FNR (False Negative Rate): most critical metric — a missed defect
    means an unsafe component reaches production.
  - FPR (False Positive Rate): causes unnecessary rework/cost.
  - Cross-entropy loss: quantifies probability calibration.

Regression metrics (for grain size prediction):
  - MSE, RMSE, MAE: error magnitude in different units.
  - MAPE: relative error in percent.
  - R²: fraction of variance explained by the model.
"""

from typing import NamedTuple

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class ClassificationMetrics(NamedTuple):
    """Classification results — FNR is the primary metric for PCG detection."""
    fnr: float
    fpr: float
    loss: float


class RegressionMetrics(NamedTuple):
    """Regression results for grain size prediction."""
    mse: float
    rmse: float
    mae: float
    mape: float
    r2: float


def compute_cls_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> ClassificationMetrics:
    """Compute FNR, FPR and cross-entropy loss for binary classification.

    Parameters
    ----------
    y_true  : ground truth labels (0 or 1)
    y_pred  : predicted labels (0 or 1)
    y_proba : predicted probability of the positive class (float in [0, 1])
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # max(..., 1) guards against division by zero when a class has 0 samples
    fnr = fn / max(fn + tp, 1)
    fpr = fp / max(fp + tn, 1)
    loss = log_loss(y_true, y_proba)

    return ClassificationMetrics(fnr=fnr, fpr=fpr, loss=loss)


def compute_reg_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> RegressionMetrics:
    """Compute MSE, RMSE, MAE, MAPE and R² for regression."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)

    # MAPE: mean absolute percentage error.
    # max(|y_true|, 1e-12) avoids division by zero on any zero-valued targets.
    mape = float(
        np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-12))) * 100
    )
    r2 = r2_score(y_true, y_pred)

    return RegressionMetrics(mse=mse, rmse=rmse, mae=mae, mape=mape, r2=r2)


def print_reg_metrics(metrics: RegressionMetrics, model_name: str) -> None:
    """Pretty-print regression metrics to stdout."""
    print(f"\n{model_name} regression results:")
    print(f"  MSE  = {metrics.mse:.4f}")
    print(f"  RMSE = {metrics.rmse:.4f}")
    print(f"  MAE  = {metrics.mae:.4f}")
    print(f"  MAPE = {metrics.mape:.4f}%")
    print(f"  R²   = {metrics.r2:.4f}")
