import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, average_precision_score,
                             roc_auc_score, log_loss, brier_score_loss)


def score_output(model, task, X):
    return model.predict(X) if task == "reg" else model.predict_proba(X)[:, 1]


def primary_loss(task, y, s):
    """Lower is better. Regression: MAE. Classification: 1 - average precision (PR-AUC)."""
    if task == "reg":
        return float(mean_absolute_error(y, s))
    return float(1 - average_precision_score(y, s))


def full_metrics(task, y, s):
    if task == "reg":
        return dict(MAE=float(mean_absolute_error(y, s)), RMSE=float(np.sqrt(mean_squared_error(y, s))),
                    R2=float(r2_score(y, s)))
    p = np.clip(s, 1e-7, 1 - 1e-7)
    ece = expected_calibration_error(y, p)
    return dict(PR_AUC=float(average_precision_score(y, p)), ROC_AUC=float(roc_auc_score(y, p)),
                LogLoss=float(log_loss(y, p)), Brier=float(brier_score_loss(y, p)), ECE=ece)


def expected_calibration_error(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)
