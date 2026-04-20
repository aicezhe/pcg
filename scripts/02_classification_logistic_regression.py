"""
Classification — Logistic Regression
======================================
Original MATLAB function: fitclinear (Learner='logistic')
Regularisation: Ridge (L2) and Lasso (L1).
Lambda range: logspace(-6, 1, 25)

Sklearn uses C = 1/lambda, so a small lambda = weak regularisation
(large C), and a large lambda = strong regularisation (small C).

Fix vs the original MATLAB version: the original code ran a third
"ridge (2)" sweep that was a duplicate of the first ridge sweep.
This implementation only runs Ridge and Lasso (no duplicate).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.config import RANDOM_STATE
from src.data import load_classification_data
from src.metrics import compute_cls_metrics
from src.plots import plot_metric_vs_lambda


REGULARISATIONS = ["l2", "l1"]
REGULARISATION_LABELS = ["Ridge", "Lasso"]
LAMBDAS = np.logspace(-6, 1, 25)


def main() -> None:
    X_train, X_test, y_train, y_test = load_classification_data()

    loss = np.full((len(LAMBDAS), len(REGULARISATIONS)), np.nan)
    fnr = np.full((len(LAMBDAS), len(REGULARISATIONS)), np.nan)
    fpr = np.full((len(LAMBDAS), len(REGULARISATIONS)), np.nan)

    for k, reg in enumerate(REGULARISATIONS):
        # L1 requires the 'saga' solver; L2 works with faster 'lbfgs'
        solver = "saga" if reg == "l1" else "lbfgs"

        for i, lam in enumerate(LAMBDAS):
            C = 1.0 / (lam + 1e-12)   # sklearn uses inverse regularisation
            model = LogisticRegression(
                penalty=reg,
                C=C,
                solver=solver,
                max_iter=1000,
                random_state=RANDOM_STATE,
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics = compute_cls_metrics(y_test, y_pred, y_proba)
            fnr[i, k] = metrics.fnr
            fpr[i, k] = metrics.fpr
            loss[i, k] = metrics.loss

    # ─── Plot each metric vs lambda ───────────────────────────────
    plot_metric_vs_lambda(
        LAMBDAS, fnr, REGULARISATION_LABELS,
        metric_name="FNR",
        suptitle="Logistic Regression — FNR",
        color="C0",
        filename="02_logreg_fnr.png",
    )
    plot_metric_vs_lambda(
        LAMBDAS, fpr, REGULARISATION_LABELS,
        metric_name="FPR",
        suptitle="Logistic Regression — FPR",
        color="orange",
        filename="02_logreg_fpr.png",
    )
    plot_metric_vs_lambda(
        LAMBDAS, loss, REGULARISATION_LABELS,
        metric_name="Loss",
        suptitle="Logistic Regression — Cross-Entropy Loss",
        color="red",
        filename="02_logreg_loss.png",
    )


if __name__ == "__main__":
    main()
