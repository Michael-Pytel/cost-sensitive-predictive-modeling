import numpy as np


def sweep_cutoff(y_true, p, n_vars, cap=1000):
    """Find N ≤ cap that maximises Score = TP*10 - FP*5 - n_vars*200."""
    order = np.argsort(-p)
    best_s, best_n = -np.inf, 0
    for n in range(1, min(cap, len(p)) + 1):
        idx = order[:n]
        tp = int(np.sum(y_true[idx] == 1))
        s = tp * 10 - (n - tp) * 5 - n_vars * 200
        if s > best_s:
            best_s, best_n = s, n
    if best_n == 0:
        return int(best_s), 0, 0.0, 0, 0
    idx_best = order[:best_n]
    tp_best  = int(np.sum(y_true[idx_best] == 1))
    thr_best = float(p[order[best_n - 1]])
    return int(best_s), best_n, thr_best, tp_best, best_n - tp_best


def score_at_thr(y_true, p, n_vars, thr=1 / 3, cap=1000):
    """Score using a fixed probability threshold (for comparison)."""
    order = np.argsort(-p)
    mask  = p[order] > thr
    sel   = order[mask][:cap]
    tp    = int(np.sum(y_true[sel] == 1))
    fp    = len(sel) - tp
    return tp * 10 - fp * 5 - n_vars * 200, tp, fp, len(sel)


def n_vars_used(model, coef_tol=1e-8):
    """Count features actually used by a fitted model (for NoVariables cost)."""
    if hasattr(model, "coef_"):
        return int(np.sum(np.abs(model.coef_[0]) > coef_tol))
    if hasattr(model, "feature_importances_"):
        return int(np.sum(model.feature_importances_ > 0))
    # CalibratedClassifierCV wraps an estimator
    if hasattr(model, "calibrated_classifiers_"):
        inner = model.calibrated_classifiers_[0].estimator
        return n_vars_used(inner, coef_tol)
    raise ValueError(f"Cannot determine n_vars_used for {type(model)}")
