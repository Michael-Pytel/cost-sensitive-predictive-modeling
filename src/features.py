import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score
from scipy.stats import ks_2samp


def l1_path_ranking(X_scaled, y, n_C=25, C_range=(-3, 1), max_iter=2000):
    """Return feature indices sorted by order of entry into the L1 path (most important first)."""
    Cs = np.logspace(*C_range, n_C)
    entry = np.full(X_scaled.shape[1], np.inf)
    for C in Cs:
        m = LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=max_iter)
        m.fit(X_scaled, y)
        nz = np.where(np.abs(m.coef_[0]) > 1e-8)[0]
        for j in nz:
            if entry[j] == np.inf:
                entry[j] = C
    return np.argsort(entry)  # most important first (entered at lowest C)


def mutual_info_ranking(X, y, random_state=0):
    """Return feature indices sorted by mutual information (highest first)."""
    mi = mutual_info_classif(X, y, random_state=random_state)
    return np.argsort(-mi)


def covariate_shift_auc(X_train, X_test, feat_idx=None):
    """
    Train a logistic classifier to discriminate train vs test samples.
    AUC ≈ 0.5 means no detectable shift; AUC → 1 means strong shift.
    feat_idx: optional subset of column indices (None = all).
    """
    A = X_train if feat_idx is None else X_train[:, feat_idx]
    B = X_test  if feat_idx is None else X_test[:, feat_idx]
    X = np.vstack([A, B])
    y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    clf = LogisticRegression(max_iter=1000, C=1.0)
    return float(cross_val_score(clf, X, y, cv=5, scoring="roc_auc").mean())


def per_feature_ks(X_train, X_test):
    """
    Compute KS statistic and p-value for every feature.
    Returns ks_stats (n_features,) and ks_pvals (n_features,).
    """
    n = X_train.shape[1]
    ks_stats = np.empty(n)
    ks_pvals = np.empty(n)
    for j in range(n):
        stat, pval = ks_2samp(X_train[:, j], X_test[:, j])
        ks_stats[j] = stat
        ks_pvals[j] = pval
    return ks_stats, ks_pvals
