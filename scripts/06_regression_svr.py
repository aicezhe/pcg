"""
Regression — SVR (Gaussian / RBF kernel)
=========================================
Original MATLAB function: fitrsvm
  KernelFunction='gaussian', KernelScale='auto',
  BoxConstraint=10, Epsilon=0.5

MATLAB's KernelScale='auto' corresponds most closely to
sklearn's gamma='scale' (default heuristic based on n_features
and X.var()).

Only samples WITHOUT PCG are used.
"""

import numpy as np
from sklearn.svm import SVR

from src.data import load_regression_data
from src.metrics import compute_reg_metrics, print_reg_metrics
from src.plots import plot_predicted_vs_actual, plot_standardised_residuals


def main() -> None:
    X_train, X_test, y_train, y_test = load_regression_data()

    model = SVR(
        kernel="rbf",
        gamma="scale",   # closest equivalent to MATLAB's 'auto'
        C=10,            # BoxConstraint
        epsilon=0.5,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = compute_reg_metrics(y_test, y_pred)
    print_reg_metrics(metrics, "SVR")

    # Predicted vs actual
    plot_predicted_vs_actual(
        y_test, y_pred,
        title="Regression: Grain Size (SVR) — NO PCG only",
        filename="06_svr_regression_scatter.png",
    )

    # Standardised residuals (SVR doesn't expose a train RMSE like
    # fitlm did, so we standardise by the std of test errors instead —
    # this mirrors the original MATLAB SVR script).
    err = y_pred - y_test
    residuals_std = (err - err.mean()) / max(err.std(), 1e-12)

    plot_standardised_residuals(
        y_pred, residuals_std,
        title="Outlier Analysis — SVR (TEST)",
        filename="06_svr_regression_residuals.png",
    )


if __name__ == "__main__":
    main()
