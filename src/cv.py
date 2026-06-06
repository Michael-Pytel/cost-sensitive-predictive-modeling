import copy
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from scoring import sweep_cutoff, n_vars_used

def competition_cv_evaluate(model, X, y, n_splits=5, random_state=0):
    """
    Global Out-Of-Fold (OOF) CV evaluation.
    Matches the exact logic of first_diagnostics.py and diagnostics_trees.py.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    m_clone = copy.deepcopy(model)
    
    p_oof = cross_val_predict(m_clone, X, y, cv=skf, method='predict_proba', n_jobs=-1)[:, 1]
    
    m_clone.fit(X, y)
    nv = n_vars_used(m_clone)
    
    score, N, thr, tp, fp = sweep_cutoff(y, p_oof, nv, cap=1000)

    return {
        "score": int(score),
        "N": int(N),
        "threshold": float(thr),
        "tp": int(tp),
        "fp": int(fp),
        "n_vars": int(nv)
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
    print(f"lr_l1  score={res['score']}  N={res['N']}  thr={res['threshold']:.3f}  "
          f"TP={res['tp']}  FP={res['fp']}  n_vars={res['n_vars']}")
