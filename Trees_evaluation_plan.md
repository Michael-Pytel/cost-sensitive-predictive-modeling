# Trees Evaluation Plan

This document describes the pipeline for investigating tree-based models on the given 
cost-sensitive task.

---

## Models

Four tree families were investigated:

| Model | sklearn / library |
|-------|-------------------|
| **Random Forest** | `RandomForestClassifier` |
| **Extra Trees** | `ExtraTreesClassifier` |
| **XGBoost** | `XGBClassifier` |
| **LightGBM** | `LGBMClassifier` |

Trees were used in two roles:

1. **Feature importance** — scoring and ranking features (repeated random search over hyperparameters, importance aggregated across runs).
2. **Predictive model** — training on the **reduced** feature set (493 features after MI filtering), with top-k feature selection and hyperparameter tuning.

Related artifacts:

- `notebooks/03_trees_feature_importance_analysis.ipynb` — importance scores and aggregated rankings
- `diagnostics_trees.py` — random search over models on reduced data
- `data/x_train_scaled_reduced_after_mi.csv`, `data/x_test_scaled_reduced_after_mi.csv` — reduced feature matrices
- `data/extra_feature_importance`, `data/lgbm_feature_importance`, `data/rf_feature_importance`, `data/xgb_feature_importance` - calculated feature importances
- `data/random_search` - random search of final model configuration



---

## 1. Feature preparation: Mutual Information

Correlated or redundant features dilute tree feature importances, so before ranking and modeling we computed pairwise MI for all 500 features ([`mutual_information/`](mutual_information/), `mi_results.csv`). For pairs with MI > 1 we dropped the feature with lower MI to the target (random tie-break); this removed **7** features — V32, V175, V199, V265, V329, V345, V416 — yielding **493** columns saved as `data/x_train_scaled_reduced_after_mi.csv` and `data/x_test_scaled_reduced_after_mi.csv` - after scaling applied. All tree work below uses this reduced set.

---

## 2. Feature importance rankings

[`src/model_generator.py`](src/model_generator.py) (`ModelGenerator`) samples random hyperparameter configs for RF, ExtraTrees, XGBoost, and LightGBM. In [`notebooks/03_trees_feature_importance_analysis.ipynb`](notebooks/03_trees_feature_importance_analysis.ipynb) we run **200** random configs per model type, average feature importances across those 200 runs, and save per-model rankings under `data/{rf,extra,lgbm,xgb}_feature_importance/`. We also build a combined **`ranking_total`** by averaging normalized importances across all **800** runs (200 × 4 models). These per-model and aggregate rankings are then used as the feature order for model training in [`diagnostics_trees.py`](diagnostics_trees.py) (top-k from `ranking_total`, `ranking_rf`, etc.).

---

## 3. Model random search and scoring

[`diagnostics_trees.py`](diagnostics_trees.py) samples a random hyperparameter config (`ModelGenerator`), then tries **k = 1…20** top features from the chosen ranking. For each (config, k) it computes a competition **score** as follows:

1. **OOF probabilities** — 5-fold stratified CV on all 5000 train rows; `cross_val_predict` stitches out-of-fold `predict_proba` into one vector of length 5000.
2. **Contact sweep** — sort customers by descending probability; try **N = 1…1000** contacted customers and pick **N** that maximizes  
   **Score = TP×10 − FP×5 − k×200**  
   (`sweep_cutoff` in [`src/scoring.py`](src/scoring.py)). Here **k** is the number of features used (explicit top-k), and **cap = 1000** matches the submission limit (1000 out
 of 5000).
3. **Best k per trial** — keep the k (and its N, threshold, TP, FP) with the highest score.

Each trial is appended to [`data/random_search/`](data/random_search/) as `{model_type}_{ranking}_random_search.csv` (e.g. `rf_ranking_rf_random_search.csv`), with hyperparameters, feature list, `score`, `k`, `N`, `threshold`, `tp`, and `fp` — enough to refit the best config on full train and build a test submission.

