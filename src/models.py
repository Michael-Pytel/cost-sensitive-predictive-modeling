from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


def get_models():
    """
    Return a dict of {name: unfitted_estimator}.
    Tree models receive all 500 features; n_vars_used() in scoring.py
    counts only those with non-zero importance to apply the NoVariables cost.
    """
    models = {
        "lr_l1": LogisticRegression(
            penalty="l1", solver="saga", C=0.1, max_iter=2000, random_state=0
        ),
        "lr_l2": LogisticRegression(
            penalty="l2", C=1.0, max_iter=2000, random_state=0
        ),
        "rf": RandomForestClassifier(
            n_estimators=300, max_features="sqrt", random_state=0, n_jobs=-1
        ),
        "gbm": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=0
        ),
    }
    if _HAS_LGBM:
        models["lgbm"] = LGBMClassifier(
            n_estimators=300, num_leaves=31, learning_rate=0.05,
            random_state=0, verbose=-1
        )
    if _HAS_XGB:
        models["xgb"] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=0,
            eval_metric="logloss", verbosity=0
        )
    return models
