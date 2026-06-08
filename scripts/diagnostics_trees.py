from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from model_generator import ModelGenerator
from scoring import sweep_cutoff

DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "random_search"
MODEL_TYPES = ("rf", "extra", "lgbm", "xgb")
RANKING_CHOICES = ("ranking_total", "ranking_rf", "ranking_extra", "ranking_lgbm", "ranking_xgb")


# ---------------------------------------------------------------------------
# Rankings
# ---------------------------------------------------------------------------

def _load_model_ranking(model_type: str) -> pd.DataFrame:
    """
    Load the feature importance ranking for a given model type.
    Can be one of the following: ranking_total, ranking_rf, ranking_extra, ranking_lgbm, ranking_xgb.
    """
    path = (
        DATA_DIR
        / f"{model_type}_feature_importance"
        / f"{model_type}_model_feature_importance_mi_reduced.csv"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Ranking file not found: {path}\n"
            f"Run notebook 03_trees_feature_importance_analysis.ipynb first."
        )
    long_df = pd.read_csv(path)
    summary = (
        long_df.groupby("feature_name", as_index=False)["importance"]
        .agg(importance_mean="mean", importance_std="std")
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    total = summary["importance_mean"].sum()
    if total > 0:
        summary["importance_mean"] /= total
        summary["importance_std"] /= total
    summary["rank"] = np.arange(1, len(summary) + 1)
    return summary


def load_ranking(name: str) -> list[str]:
    """Return ordered feature names (best first)."""
    if name == "ranking_total":
        parts = []
        for mt in MODEL_TYPES:
            s = _load_model_ranking(mt)[["feature_name", "importance_mean"]].copy()
            s = s.rename(columns={"importance_mean": f"{mt}_importance_mean"})
            parts.append(s)
        merged = parts[0]
        for part in parts[1:]:
            merged = merged.merge(part, on="feature_name", how="outer")
        mean_cols = [f"{mt}_importance_mean" for mt in MODEL_TYPES]
        merged["importance_mean"] = merged[mean_cols].mean(axis=1)
        merged = merged.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    elif name == "ranking_rf":
        merged = _load_model_ranking("rf")
    elif name == "ranking_extra":
        merged = _load_model_ranking("extra")
    elif name == "ranking_lgbm":
        merged = _load_model_ranking("lgbm")
    elif name == "ranking_xgb":
        merged = _load_model_ranking("xgb")
    else:
        raise ValueError(f"Unknown ranking: {name}")
    return merged["feature_name"].tolist()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_train() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """
    Load the train data, after mutual information reduction and scaling.
    """
    X = pd.read_csv(DATA_DIR / "processed" / "x_train_scaled_reduced_after_mi.csv")
    y = np.loadtxt(DATA_DIR / "y_train.txt", skiprows=1).astype(int)
    return X, y, X.columns.tolist()


# ---------------------------------------------------------------------------
# Model + evaluation
# ---------------------------------------------------------------------------

def build_model(model_type: str, params: dict, random_seed: int):
    """
    Build a model of a given type with given parameters and random seed.
    """
    gen = ModelGenerator(config_type="random", random_seed=random_seed)
    cls = gen.model_map[model_type]
    if model_type == "xgb":
        return cls(**params, seed=random_seed, n_jobs=-1, eval_metric="logloss")
    m = cls(**params, random_state=random_seed)
    if hasattr(m, "n_jobs"):
        m.set_params(n_jobs=-1)
    return m


def diagnostics_evaluate(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_vars: int,
    cv: int = 5,
    random_state: int = 0,
    cap: int = 1000,
) -> dict:
    """
    OOF proba on full train + one global sweep_cutoff (first_diagnostics.py protocol).
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    p_oof = cross_val_predict(
        copy.deepcopy(model),
        X,
        y,
        cv=skf,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    score, N, thr, tp, fp = sweep_cutoff(y, p_oof, n_vars=n_vars, cap=cap)
    return {
        "score": int(score),
        "N": int(N),
        "threshold": float(thr),
        "tp": int(tp),
        "fp": int(fp),
        "p_oof_mean": float(p_oof.mean()),
    }


def search_best_k(
    model,
    X_df: pd.DataFrame,
    y: np.ndarray,
    ranking: list[str],
    k_min: int = 1,
    k_max: int = 20,
    cv: int = 5,
    random_state: int = 0,
) -> dict | None:
    """
    Search for the best k (number of features) for a given model, data, and ranking.
    """
    best = None
    for k in range(k_min, k_max + 1):
        feats = ranking[:k]
        missing = [f for f in feats if f not in X_df.columns]
        if missing:
            raise ValueError(f"Features not in X_train: {missing[:5]}")
        X_sub = X_df[feats].to_numpy()
        ev = diagnostics_evaluate(model, X_sub, y, n_vars=k, cv=cv, random_state=random_state)
        row = {"k": k, "features": feats, **ev}
        if best is None or row["score"] > best["score"]:
            best = row
    return best


# ---------------------------------------------------------------------------
# Random search loop
# ---------------------------------------------------------------------------

RESULT_COLUMNS = [
    "trial_id",
    "timestamp_utc",
    "model_type",
    "ranking",
    "random_seed",
    "score",
    "k",
    "N",
    "threshold",
    "tp",
    "fp",
    "p_oof_mean",
    "features",
    "hyperparameters",
]


def append_trial(csv_path: Path, row: dict) -> None:
    """
    Append a trial to the results CSV file.
    """
    df = pd.DataFrame([row], columns=RESULT_COLUMNS)
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=header, index=False)


def run_random_search(
    model_type: str,
    ranking_name: str,
    n_trials: int = 50,
    k_max: int = 20,
    random_seed: int = 0,
    cv: int = 5,
) -> Path:
    """
    Main function to run the random search loop.
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{model_type}_{ranking_name}_random_search.csv"

    X_df, y, _ = load_train()
    ranking = load_ranking(ranking_name)
    gen = ModelGenerator(config_type="random", random_seed=random_seed)

    print(f"model={model_type}  ranking={ranking_name}  trials={n_trials}  k=1..{k_max}")
    print(f"output: {out_path}")

    for trial in range(1, n_trials + 1):
        t0 = time.time()
        trial_seed = random_seed + trial
        gen.random_seed = trial_seed
        _, params, _ = gen.generate(model_type)
        model = build_model(model_type, params, trial_seed)

        best = search_best_k(
            model,
            X_df,
            y,
            ranking,
            k_max=k_max,
            cv=cv,
            random_state=random_seed,
        )
        if best is None:
            continue

        row = {
            "trial_id": trial,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_type": model_type,
            "ranking": ranking_name,
            "random_seed": trial_seed,
            "score": best["score"],
            "k": best["k"],
            "N": best["N"],
            "threshold": best["threshold"],
            "tp": best["tp"],
            "fp": best["fp"],
            "p_oof_mean": best["p_oof_mean"],
            "features": json.dumps(best["features"]),
            "hyperparameters": json.dumps(params),
        }
        append_trial(out_path, row)

        dt = time.time() - t0
        print(
            f"trial {trial:3d}/{n_trials}  score={best['score']:5d}  "
            f"k={best['k']:2d}  N={best['N']:4d}  thr={best['threshold']:.4f}  "
            f"TP={best['tp']}  FP={best['fp']}  ({dt:.1f}s)"
        )

    if out_path.exists():
        results = pd.read_csv(out_path)
        best_row = results.loc[results["score"].idxmax()]
        print("\n=== BEST OVER ALL TRIALS ===")
        print(best_row.to_string())
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """
    Parse command line arguments, such as model type, ranking, number of trials, etc.
    """
    p = argparse.ArgumentParser(description="Tree random search with diagnostics OOF scoring")
    p.add_argument("--model-type", required=True, choices=MODEL_TYPES)
    p.add_argument("--ranking", required=True, choices=RANKING_CHOICES)
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--k-max", type=int, default=20)
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--cv", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_random_search(
        model_type=args.model_type,
        ranking_name=args.ranking,
        n_trials=args.n_trials,
        k_max=args.k_max,
        random_seed=args.random_seed,
        cv=args.cv,
    )