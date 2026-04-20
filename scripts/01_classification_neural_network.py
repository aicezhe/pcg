"""
Classification — Neural Network (MLP)
======================================
Original MATLAB function: fitcnet
Task: predict PCG presence (binary: yes / no) from extrusion process
parameters.

Grid search over:
  - number of neurons in each hid den layer (1..8 × 1..8)
  - activation function (ReLU / Sigmoid / tanh)

Critical metric: FNR (False Negative Rate).
Missing a PCG defect means an unsafe component reaches production,
so minimising FNR is more important than minimising FPR.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier

from src.config import RANDOM_STATE
from src.data import load_classification_data
from src.metrics import compute_cls_metrics
from src.plots import plot_heatmap_grid_3d


# Activations compared in the grid search
ACTIVATIONS = ["relu", "logistic", "tanh"]
ACTIVATION_LABELS = ["ReLU", "Sigmoid", "tanh"]
MAX_NEURONS = 8   # grid: 1..8 neurons per layer


def main() -> None:
    X_train, X_test, y_train, y_test = load_classification_data()

    # Arrays to hold results: shape (neurons_layer1, neurons_layer2, activation)
    loss = np.full((MAX_NEURONS, MAX_NEURONS, 3), np.nan)
    fnr = np.full((MAX_NEURONS, MAX_NEURONS, 3), np.nan)
    fpr = np.full((MAX_NEURONS, MAX_NEURONS, 3), np.nan)

    for k, act in enumerate(ACTIVATIONS):
        print(f"Activation: {ACTIVATION_LABELS[k]}")
        for i in range(1, MAX_NEURONS + 1):
            for j in range(1, MAX_NEURONS + 1):
                model = MLPClassifier(
                    hidden_layer_sizes=(i, j),
                    activation=act,
                    max_iter=500,
                    random_state=RANDOM_STATE,
                )
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]

                metrics = compute_cls_metrics(y_test, y_pred, y_proba)
                fnr[i - 1, j - 1, k] = metrics.fnr
                fpr[i - 1, j - 1, k] = metrics.fpr
                loss[i - 1, j - 1, k] = metrics.loss

    # ─── Plot FNR grid ────────────────────────────────────────────
    # vmax=None → matplotlib auto-scales the colour range to the actual data.
    # This makes differences between architectures and activations visible
    # (FNR varies in a narrow band ~0.16-0.30, so a fixed vmax=0.15 would
    # saturate everything to the same yellow colour).
    plot_heatmap_grid_3d(
        fnr,
        titles=[f"FNR, PCG, {lbl}" for lbl in ACTIVATION_LABELS],
        suptitle="Neural Network — FNR Grid Search",
        vmax=None,
        filename="01_nn_fnr.png",
    )

    # ─── Plot FPR grid ────────────────────────────────────────────
    # FPR stays low (~0.03), so vmax=0.15 keeps the scale comparable
    # to FNR's previous default while still showing the variation.
    plot_heatmap_grid_3d(
        fpr,
        titles=[f"FPR, PCG, {lbl}" for lbl in ACTIVATION_LABELS],
        suptitle="Neural Network — FPR Grid Search",
        vmax=0.15,
        filename="01_nn_fpr.png",
    )

    # ─── Plot Loss grid ───────────────────────────────────────────
    plot_heatmap_grid_3d(
        loss,
        titles=[f"Loss, {lbl}" for lbl in ACTIVATION_LABELS],
        suptitle="Neural Network — Cross-Entropy Loss Grid Search",
        filename="01_nn_loss.png",
    )


if __name__ == "__main__":
    main()
