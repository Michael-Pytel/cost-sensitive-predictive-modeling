import copy
import numpy as np
from sklearn.model_selection import StratifiedKFold
from scoring import sweep_cutoff, n_vars_used


def competition_cv_evaluate(model, X, y, n_splits=5, random_state=0):
    """
    5-fold stratified CV (each val fold = 1000 obs for 5000-sample train set).

    For each fold:
      - fit a fresh copy of model on the 4000-obs train split
      - predict_proba on the 1000-obs val split
      - count n_vars_used on the fitted model
      - run sweep_cutoff to find the optimal N ≤ 1000 that maximises
        Score = TP*10 - FP*5 - n_vars*200

    Returns
    -------
    results : dict with keys
        "folds"   : list of per-fold dicts (score, N, threshold, tp, fp, n_vars)
        "scores"  : np.array of per-fold scores
        "mean_score", "std_score"
        "mean_N",    "std_N"
        "mean_thr",  "std_thr"
        "mean_nvars","std_nvars"
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        m = copy.deepcopy(model)
        m.fit(X_tr, y_tr)

        p_val = m.predict_proba(X_val)[:, 1]
        nv    = n_vars_used(m)
        score, N, thr, tp, fp = sweep_cutoff(y_val, p_val, nv)

        folds.append({"score": score, "N": N, "threshold": thr,
                      "tp": tp, "fp": fp, "n_vars": nv})

    scores  = np.array([f["score"]     for f in folds])
    Ns      = np.array([f["N"]         for f in folds])
    thrs    = np.array([f["threshold"] for f in folds])
    nvars   = np.array([f["n_vars"]    for f in folds])

    return {
        "folds":       folds,
        "scores":      scores,
        "mean_score":  float(scores.mean()),
        "std_score":   float(scores.std()),
        "mean_N":      float(Ns.mean()),
        "std_N":       float(Ns.std()),
        "mean_thr":    float(thrs.mean()),
        "std_thr":     float(thrs.std()),
        "mean_nvars":  float(nvars.mean()),
        "std_nvars":   float(nvars.std()),
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    from data import load_data
    from models import get_models

    print("Loading data...")
    _, y_train, _, X_train_s, _, _ = load_data()

    models = get_models()
    print(f"Smoke-testing lr_l1 on {X_train_s.shape[0]} samples, {X_train_s.shape[1]} features")
    res = competition_cv_evaluate(models["lr_l1"], X_train_s, y_train)
    print(f"lr_l1  mean_score={res['mean_score']:.0f} ± {res['std_score']:.0f}  "
          f"mean_N={res['mean_N']:.0f}  mean_nvars={res['mean_nvars']:.1f}")
    for i, f in enumerate(res["folds"]):
        print(f"  fold {i+1}: score={f['score']}  N={f['N']}  thr={f['threshold']:.3f}  "
              f"TP={f['tp']}  FP={f['fp']}  n_vars={f['n_vars']}")
