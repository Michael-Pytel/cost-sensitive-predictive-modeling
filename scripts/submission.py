"""
submission.py — Final model training and test-set submission generation.

Trains the best ExtraTrees configuration (k=6, Optuna-tuned) on the full
training set and produces the two submission files required by the competition:
  * <IDs>_obs.txt  — up to 1000 test customer indices (1-indexed)
  * <IDs>_vars.txt — indices of features used (1-indexed)

Usage:
    python submission.py --ids 123456_654321_111111

The script also prints OOF (out-of-fold) score on training data as a sanity
check before generating test predictions.
"""

import argparse
import json
import copy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict


# ---------------------------------------------------------------------------
# Configuration — best model from exhaustive search + Optuna (notebook 05)
# ---------------------------------------------------------------------------

FEATURES = ["V176", "V255", "V191", "V160", "V380", "V342"]

HYPERPARAMETERS = {
    "n_estimators": 214,
    "max_depth": 12,
    "min_samples_split": 9,
    "min_samples_leaf": 8,
    "criterion": "gini",
    "class_weight": "balanced",
}

RANDOM_STATE = 0
CV_FOLDS = 5
CAP = 1000  # max contacts allowed


# ---------------------------------------------------------------------------
# Scoring (self-contained copy to avoid import issues)
# ---------------------------------------------------------------------------

def sweep_cutoff(y_true, p, n_vars, cap=CAP):
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
    tp_best = int(np.sum(y_true[order[:best_n]] == 1))
    thr_best = float(p[order[best_n - 1]])
    return int(best_s), best_n, thr_best, tp_best, best_n - tp_best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate competition submission files."
    )
    parser.add_argument(
        "--ids",
        type=str,
        required=True,
        help="Team student IDs joined by underscores, e.g. 123456_654321_111111",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to the data directory (default: data/)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=".",
        help="Directory for output files (default: current directory)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Loading data...")
    X_train_full = pd.read_csv(data_dir / "processed" / "x_train_scaled_reduced_after_mi.csv")
    y_train = np.loadtxt(data_dir / "y_train.txt", skiprows=1).astype(int)
    X_test_full = pd.read_csv(data_dir / "processed" / "x_test_scaled_reduced_after_mi.csv")

    X_train = X_train_full[FEATURES].to_numpy()
    X_test = X_test_full[FEATURES].to_numpy()
    k = len(FEATURES)

    print(f"  Train: {X_train.shape}   Test: {X_test.shape}")
    print(f"  Features: {FEATURES}")
    print(f"  Positive rate (train): {y_train.mean():.3f}")

    # ------------------------------------------------------------------
    # 2. OOF sanity check
    # ------------------------------------------------------------------
    print("\nOOF sanity check (5-fold CV on training data)...")
    model_oof = ExtraTreesClassifier(
        **HYPERPARAMETERS, random_state=RANDOM_STATE, n_jobs=-1
    )
    skf = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    p_oof = cross_val_predict(
        copy.deepcopy(model_oof), X_train, y_train,
        cv=skf, method="predict_proba", n_jobs=1
    )[:, 1]

    score, N_oof, thr, tp, fp = sweep_cutoff(y_train, p_oof, n_vars=k)
    print(f"  OOF score = {score}")
    print(f"  N = {N_oof}   TP = {tp}   FP = {fp}   threshold = {thr:.4f}")
    print(f"  precision@N = {tp / N_oof:.1%}")

    # ------------------------------------------------------------------
    # 3. Train final model on ALL training data
    # ------------------------------------------------------------------
    print("\nTraining final model on full training set...")
    model_final = ExtraTreesClassifier(
        **HYPERPARAMETERS, random_state=RANDOM_STATE, n_jobs=-1
    )
    model_final.fit(X_train, y_train)

    p_test = model_final.predict_proba(X_test)[:, 1]
    print(f"  Test proba: mean={p_test.mean():.3f}  median={np.median(p_test):.3f}")
    print(f"  Fraction p > 0.5: {(p_test > 0.5).mean():.3f}")

    # ------------------------------------------------------------------
    # 4. Select top-N customers to contact
    # ------------------------------------------------------------------
    # Use the OOF-optimal N (capped at 1000)
    # Since train/test distributions are identical (no covariate shift),
    # we use the same N found during CV.
    N_contact = min(N_oof, CAP)

    order = np.argsort(-p_test)
    selected_indices = order[:N_contact]

    # Convert to 1-indexed (competition format)
    obs_1indexed = (selected_indices + 1).tolist()

    # Feature indices: extract original column numbers from feature names
    # V176 -> index 176, etc.
    var_indices = sorted([int(f[1:]) for f in FEATURES])

    print(f"\n  Contacting {N_contact} customers")
    print(f"  Variables used: {var_indices} (k={k}, cost={k * 200})")

    # ------------------------------------------------------------------
    # 5. Write submission files
    # ------------------------------------------------------------------
    obs_path = out_dir / f"{args.ids}_obs.txt"
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
    # 6. Summary
    # ------------------------------------------------------------------
    summary = {
        "team_ids": args.ids,
        "model": "ExtraTreesClassifier",
        "features": FEATURES,
        "feature_indices_1indexed": var_indices,
        "k": k,
        "hyperparameters": HYPERPARAMETERS,
        "oof_score": score,
        "oof_N": N_oof,
        "oof_tp": tp,
        "oof_fp": fp,
        "oof_threshold": round(thr, 4),
        "n_test_contacted": N_contact,
        "random_state": RANDOM_STATE,
    }

    summary_path = out_dir / "submission_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")

    print("\n" + "=" * 50)
    print("SUBMISSION READY")
    print(f"  Expected score range: ~5400-5700 (OOF = {score})")
    print("=" * 50)


if __name__ == "__main__":
    main()
