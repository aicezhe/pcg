"""
Regression — Linear Model (OLS)
================================
Original MATLAB function: fitlm
Task: predict grain size using ordinary least-squares linear regression.

Only samples WITHOUT PCG are used (same physical constraint as for the
neural-network regressor).

In addition to the standard regression metrics, this script plots
standardised residuals to flag potential outliers and check for
systematic bias. Points outside the ±2 band deviate more than two
standard deviations from the typical training error.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from src.data import load_regression_data
from src.metrics import compute_reg_metrics, print_reg_metrics
from src.plots import plot_predicted_vs_actual, plot_standardised_residuals


def main() -> None:
    X_train, X_test, y_train, y_test = load_regression_data()

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = compute_reg_metrics(y_test, y_pred)
    print_reg_metrics(metrics, "Linear Regression")

    # Predicted vs actual
    plot_predicted_vs_actual(
        y_test, y_pred,
        title="Linear Regression — Grain Size (NO PCG only)",
        filename="05_linear_regression_scatter.png",
    )

    # Standardised residuals (using TRAIN RMSE as the scale, as in the
    # original MATLAB code): this highlights points whose error is
    # unusually large compared to the typical training error.
    y_train_pred = model.predict(X_train)
    train_rmse = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))
    residuals_std = (y_pred - y_test) / max(train_rmse, 1e-12)

    plot_standardised_residuals(
        y_pred, residuals_std,
        title="Outlier Analysis — Linear Regression (TEST)",
        filename="05_linear_regression_residuals.png",
    )


if __name__ == "__main__":
    main()
