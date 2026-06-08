"""
submission.py — Final model training and test-set submission generation.

Trains the best SVM-RBF configuration (k=5, Optuna-tuned) on the full
training set and produces the two submission files required by the project.

Usage:
    python submission.py --ids 123456_654321_111111

The script also prints the OOF (out-of-fold) score on training data as a
sanity check before generating test predictions.
"""

import argparse
import json
import copy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict


# ---------------------------------------------------------------------------
# Configuration - best model from exhaustive subset search + Optuna (nb 08)
# Loaded from svm_best_config.json if present; hardcoded values are fallback.
# ---------------------------------------------------------------------------

def _load_config(path="svm_best_config.json"):
    for p in [Path(path), Path(__file__).parent / path]:
        if p.exists():
            cfg = json.load(open(p))
            print(f"Loaded config from {p}")
            return cfg["features"], {"C": cfg["C"], "gamma": cfg["gamma"]}
    return None, None

_feats, _hp = _load_config()

FEATURES = _feats or ["V255", "V191", "V160", "V380", "V342"]

HYPERPARAMETERS = _hp or {
    "C":     0.02947,
    "gamma": 0.2616,
}

RANDOM_STATE = 1
CV_FOLDS     = 5
CAP          = 1000   

EXPECTED_OOF_MEAN = 5638  



def sweep_cutoff(y_true, scores, n_vars, cap=CAP):
    """Find N <= cap that maximises Score = TP*10 - FP*5 - n_vars*200.

    Works with both predict_proba[:,1] and decision_function output -
    sweep_cutoff is rank-based so calibration does not matter.
    """
    order = np.argsort(-scores)
    best_s, best_n = -np.inf, 0
    for n in range(1, min(cap, len(scores)) + 1):
        tp = int(np.sum(y_true[order[:n]] == 1))
        s  = tp * 10 - (n - tp) * 5 - n_vars * 200
        if s > best_s:
            best_s, best_n = s, n
    if best_n == 0:
        return int(best_s), 0, 0.0, 0, 0
    tp_best  = int(np.sum(y_true[order[:best_n]] == 1))
    thr_best = float(scores[order[best_n - 1]])
    return int(best_s), best_n, thr_best, tp_best, best_n - tp_best


def main():
    parser = argparse.ArgumentParser(
        description="Generate competition submission files."
    )
    parser.add_argument(
        "--ids", type=str, required=True,
        help="Team student IDs joined by underscores, e.g. 123456_654321_111111",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Path to the data directory (default: data/)",
    )
    parser.add_argument(
        "--out-dir", type=str, default=".",
        help="Directory for output files (default: current directory)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    X_train_full = pd.read_csv(
        data_dir / "processed" / "x_train_scaled_reduced_after_mi.csv")
    y_train = np.loadtxt(data_dir / "y_train.txt", skiprows=1).astype(int)
    X_test_full = pd.read_csv(
        data_dir / "processed" / "x_test_scaled_reduced_after_mi.csv")

    X_train = X_train_full[FEATURES].to_numpy()
    X_test  = X_test_full[FEATURES].to_numpy()
    k       = len(FEATURES)

    print(f"  Train: {X_train.shape}   Test: {X_test.shape}")
    print(f"  Features ({k}): {FEATURES}")
    print(f"  Positive rate (train): {y_train.mean():.3f}")

    # ------------------------------------------------------------------
    # OOF sanity check (decision_function, consistent with training)
    # ------------------------------------------------------------------
    print("\nOOF sanity check (5-fold CV on training data)...")
    model_oof = SVC(
        kernel="rbf", **HYPERPARAMETERS,
        probability=False, random_state=RANDOM_STATE,
    )
    skf = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    df_oof = cross_val_predict(
        copy.deepcopy(model_oof), X_train, y_train,
        cv=skf, method="decision_function", n_jobs=1,
    )

    score, N_oof, thr, tp, fp = sweep_cutoff(y_train, df_oof, n_vars=k)
    print(f"  OOF score = {score}  (expected ≈ {EXPECTED_OOF_MEAN})")
    print(f"  N = {N_oof}   TP = {tp}   FP = {fp}   threshold = {thr:.4f}")
    print(f"  precision@N = {tp / N_oof:.1%}")

    # ------------------------------------------------------------------
    # Train final model on ALL training data
    # ------------------------------------------------------------------
    print("\nTraining final model on full training set...")
    model_final = SVC(
        kernel="rbf", **HYPERPARAMETERS,
        probability=False, random_state=RANDOM_STATE,
    )
    model_final.fit(X_train, y_train)

    # Rank test customers by decision function (higher = more likely to buy)
    df_test = model_final.decision_function(X_test)
    print(f"  decision_function: mean={df_test.mean():.3f}  "
          f"median={np.median(df_test):.3f}")

    # ------------------------------------------------------------------
    # Select top-N customers to contact
    # ------------------------------------------------------------------
    N_contact = min(N_oof, CAP)
    order = np.argsort(-df_test)
    selected_indices = order[:N_contact]

    # Convert to 1-indexed
    obs_1indexed = (selected_indices + 1).tolist()

    # Feature indices: extract original column numbers from feature names
    var_indices = sorted([int(f[1:]) for f in FEATURES])

    print(f"\n  Contacting {N_contact} customers")
    print(f"  Variables used: {var_indices}  (k={k}, cost={k * 200})")

    # ------------------------------------------------------------------
    # Write submission files
    # ------------------------------------------------------------------
    obs_path  = out_dir / f"{args.ids}_obs.txt"
    vars_path = out_dir / f"{args.ids}_vars.txt"

    with open(obs_path, "w") as f:
        for idx in obs_1indexed:
            f.write(f"{idx}\n")

    with open(vars_path, "w") as f:
        for v in var_indices:
            f.write(f"{v}\n")

    print(f"\n  Saved: {obs_path}  ({len(obs_1indexed)} indices)")
    print(f"  Saved: {vars_path}  ({len(var_indices)} variables)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary = {
        "team_ids":              args.ids,
        "model":                 "SVC (RBF kernel)",
        "features":              FEATURES,
        "feature_indices_1indexed": var_indices,
        "k":                     k,
        "hyperparameters":       {**HYPERPARAMETERS, "kernel": "rbf"},
        "oof_score":             score,
        "oof_N":                 N_oof,
        "oof_tp":                tp,
        "oof_fp":                fp,
        "oof_threshold":         round(thr, 4),
        "n_test_contacted":      N_contact,
        "random_state":          RANDOM_STATE,
    }

    summary_path = out_dir / "submission_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")

    print("\n" + "=" * 50)
    print("SUBMISSION READY")
    print(f"  OOF score (seed 1): {score}")
    print(f"  Expected score (multi-seed mean): ~{EXPECTED_OOF_MEAN}")
    print("=" * 50)

if __name__ == "__main__":
    main()