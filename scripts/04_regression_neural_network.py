"""
Regression — Neural Network (MLP)
===================================
Original MATLAB function: fitrnet
Task: predict grain size (Dimensione Grano) — continuous target.

IMPORTANT physical constraint:
  Only samples WITHOUT PCG are used. PCG fundamentally alters
  microstructural evolution and would contaminate the regression
  (suggested by the professor; physically motivated).

Architecture: hidden_layer_sizes=(20, 10), activation='relu',
              alpha=1e-4 (L2 regularisation, i.e. lambda in MATLAB)

In addition to the standard regression metrics, this script also
computes permutation feature importance to see which inputs matter
most for grain size prediction.
"""

from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPRegressor

from src.config import RANDOM_STATE
from src.data import load_regression_data
from src.metrics import compute_reg_metrics, print_reg_metrics
from src.plots import plot_feature_importance, plot_predicted_vs_actual


def main() -> None:
    X_train, X_test, y_train, y_test = load_regression_data()

    model = MLPRegressor(
        hidden_layer_sizes=(20, 10),
        activation="relu",
        alpha=1e-4,
        max_iter=500,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = compute_reg_metrics(y_test, y_pred)
    print_reg_metrics(metrics, "Neural Network")

    # Predicted vs actual
    plot_predicted_vs_actual(
        y_test, y_pred,
        title="Regression: Grain Size (Neural Network) — NO PCG only",
        filename="04_nn_regression_scatter.png",
    )

    # Permutation feature importance
    perm = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE
    )
    plot_feature_importance(
        perm.importances_mean,
        title="Mean Permutation Importance — Grain Size (Neural Network, NO PCG)",
        filename="04_nn_feature_importance.png",
    )


if __name__ == "__main__":
    main()
