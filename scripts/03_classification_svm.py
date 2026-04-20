"""
Classification — SVM (RBF kernel)
==================================
Original MATLAB function: fitcsvm (KernelFunction='rbf')
Grid search over:
  - C (BoxConstraint): logspace(-1, 1, 5)
  - gamma (KernelScale): logspace(-1, 1, 5)

Fix vs the original: this version computes cross-entropy loss
(using predict_proba), consistent with the other classifiers —
instead of the error rate that the old code labelled as 'Loss'.
This makes the three classifiers directly comparable on the same
metric.
"""

import numpy as np
from sklearn.svm import SVC

from src.config import RANDOM_STATE
from src.data import load_classification_data
from src.metrics import compute_cls_metrics
from src.plots import plot_svm_heatmap


C_VALS = np.logspace(-1, 1, 5)
G_VALS = np.logspace(-1, 1, 5)


def main() -> None:
    X_train, X_test, y_train, y_test = load_classification_data()

    fnr = np.full((len(C_VALS), len(G_VALS)), np.nan)
    fpr = np.full((len(C_VALS), len(G_VALS)), np.nan)
    loss = np.full((len(C_VALS), len(G_VALS)), np.nan)

    for i, C in enumerate(C_VALS):
        print(f"C = {C:.3f}")
        for j, gamma in enumerate(G_VALS):
            # probability=True is needed so we can compute log_loss
            # consistently with the other classifiers.
            model = SVC(
                kernel="rbf",
                C=C,
                gamma=gamma,
                probability=True,
                random_state=RANDOM_STATE,
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics = compute_cls_metrics(y_test, y_pred, y_proba)
            fnr[i, j] = metrics.fnr
            fpr[i, j] = metrics.fpr
            loss[i, j] = metrics.loss

    plot_svm_heatmap(fnr, C_VALS, G_VALS,
                     title="FNR (RBF SVM)",
                     filename="03_svm_fnr.png")
    plot_svm_heatmap(fpr, C_VALS, G_VALS,
                     title="FPR (RBF SVM)",
                     filename="03_svm_fpr.png")
    plot_svm_heatmap(loss, C_VALS, G_VALS,
                     title="Cross-Entropy Loss (RBF SVM)",
                     filename="03_svm_loss.png")


if __name__ == "__main__":
    main()
