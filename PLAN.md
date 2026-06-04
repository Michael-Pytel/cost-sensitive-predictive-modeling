# Plan: Comprehensive Cost-Sensitive EDA + Modeling Analysis

## Context

The existing `diagnostics_mp.py` is a single-file script focused narrowly on L1-path feature ranking and a simple CV sweep. The user wants a fully restructured analysis with:
- Modern EDA (visual, interpretive, not just print statements)
- Multiple models evaluated via a proper competition-aware CV loop (1000-obs val folds = 5-fold stratified on 5000 training samples)
- Clean separation: `/src` for reusable logic, `/notebooks` for narrative + plots + tables
- Model comparison differentiated clearly in the notebook

Key data facts (from summary.json output):
- 5000 train / 5000 test, 500 features, binary classification
- Positive rate ~50% (user confirms "close to 50/50")
- **Baseline CV score: 3885** (LR-L1, 5 features) — any model scoring below this is worse than what we have
- Features used so far: V175, V32, V390, V345, V224

Scoring: `Score = TP*10 - FP*5 - NoVariables*200`, cap **up to** 1000 contacts.
The model chooses N ≤ 1000 per fold via `sweep_cutoff` — contacting fewer customers is valid and preferred when it yields a higher score.
The existing `sweep_cutoff` and `score_at_thr` in `diagnostics_mp.py` are the source of truth for scoring logic.

---

## Project Structure to Create

```
/src/
  data.py         ← data loading (Path-based, no hardcoded /mnt paths)
  scoring.py      ← sweep_cutoff, score_at_thr, n_vars_used (moved from diagnostics_mp.py)
  features.py     ← L1-path ranking, mutual info ranking, covariate shift AUC, per-feature KS
  models.py       ← model factory (LR-L1, LR-L2, RF, LightGBM, XGBoost)
  cv.py           ← competition_cv_evaluate: CV loop with sweep_cutoff scoring

/notebooks/
  01_eda.ipynb    ← EDA narrative: distributions, shift, separability, calibration
  02_models.ipynb ← Model comparison: CV competition scores, feature count vs score, plots
```

---

## `/src/data.py`

```python
DATA_DIR = Path(__file__).parent.parent / "data"

def load_data():
    X_train, y_train, X_test
    # also returns StandardScaler fitted on train
```

---

## `/src/scoring.py`

Copy `sweep_cutoff` and `score_at_thr` from `diagnostics_mp.py` verbatim.

Add `n_vars_used(model)`:
- For linear models: count of |coef| > threshold
- For tree models: count of feature importances > 0
This is critical — tree models implicitly select features and NoVariables must reflect actual usage.

---

## `/src/features.py`

- `l1_path_ranking(X_scaled, y)` — L1 path entry order (from diagnostics_mp.py)
- `mutual_info_ranking(X, y)` — sklearn mutual_info_classif, returns sorted indices
- `covariate_shift_auc(X_train, X_test, feat_idx=None)` — discriminator AUC (logistic, 5-fold)
- `per_feature_ks(X_train, X_test)` — returns array of KS stats and p-values for all 500 features

---

## `/src/models.py`

Model factory returning configured sklearn-compatible estimators:

```python
MODELS = {
    "lr_l1":  LogisticRegression(penalty='l1', solver='saga', C=0.1, max_iter=2000),
    "lr_l2":  LogisticRegression(penalty='l2', C=1.0, max_iter=2000),
    "rf":     RandomForestClassifier(n_estimators=300, max_features='sqrt', random_state=0),
    "lgbm":   LGBMClassifier(n_estimators=300, num_leaves=31, random_state=0),
    "xgb":    XGBClassifier(n_estimators=300, max_depth=4, random_state=0, eval_metric='logloss'),
}
```

Note: tree models use all 500 features as input; `n_vars_used` in scoring.py will count only features with importance > 0 to compute the NoVariables cost. This rewards naturally sparse models.

---

## `/src/cv.py`

```python
def competition_cv_evaluate(model, X, y, n_splits=5, random_state=0):
    """
    5-fold StratifiedKFold (each val fold = 1000 obs).
    For each fold:
      - fit model on 4000 train obs
      - predict_proba on 1000 val obs
      - count n_vars_used(fitted_model)
      - run sweep_cutoff(y_val, p_val, n_vars)
    Returns dict of per-fold: score, N_contacted, threshold, tp, fp, n_vars
    + summary stats (mean/std of score, N, threshold)
    """
```

Using `StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)` — this produces exactly 1000-obs val folds given 5000 training samples while preserving ~50/50 label proportion.

`sweep_cutoff` finds the best N ≤ 1000 for each fold independently — the model may contact fewer than 1000 if that improves the score (e.g., if the NoVariables cost is high, a more targeted N is preferred). N=1000 is a hard cap, not a target.

---

## `/notebooks/01_eda.ipynb`

Sections (each with plots + markdown interpretation):

1. **Data overview** — shape, positive rate, header check, table of basic stats
2. **Label distribution** — bar chart with counts and percentages
3. **Feature variance landscape** — histogram of per-feature std across 500 features; identify constant/near-constant features
4. **Covariate shift** — scatter plot of per-feature KS statistic vs train mean; highlight features with high shift; bar chart of top-20 most shifted features; PCA plot of train vs test (color-coded) on top-10 PCA components
5. **Class separability** — for the top-10 L1 features: overlapping density plots of feature distribution by class (y=0 vs y=1); combined violin/box plot
6. **Correlation structure** — heatmap of top-20 features by L1 rank; annotate with Spearman r
7. **Calibration check** — reliability diagram (calibration curve) on 5-fold CV predictions with a logistic baseline; Brier score

Each plot saved as PNG to `/notebooks/figures/`.

---

## `/notebooks/02_models.ipynb`

Sections:

1. **Feature selection comparison** — table comparing L1 path vs mutual_info top-10 features; Venn overlap
2. **CV competition score per model** — grouped bar chart: mean ± std score across 5 folds for all 5 models; table with score / N_contacted / n_vars / threshold columns
3. **Score vs features used** — line plot: for LR-L1, sweep k=1..30 and plot CV score vs k (NoVariables cost is explicit here); mark the point where adding a feature costs more than it gains
4. **N_contacted distribution per fold** — box plot per model showing how consistent the optimal N is across folds (stability indicator)
5. **Threshold stability** — violin plot of optimal threshold across 5 folds per model
6. **Baseline comparison** — horizontal reference line at score=3885 on all score plots; any model/config below baseline is marked as "worse than current" and excluded from further discussion
7. **Model selection rationale** — markdown cell with interpretation: which model beats the baseline, by how much, and whether the gain justifies any increase in NoVariables cost

---

## Files to create

- `src/data.py`
- `src/scoring.py`
- `src/features.py`
- `src/models.py`
- `src/cv.py`
- `notebooks/01_eda.ipynb`
- `notebooks/02_models.ipynb`

Do NOT modify `diagnostics_mp.py` — it stays as the original single-file diagnostic.

---

## Dependencies

All standard: `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `lightgbm`, `xgboost`, `jupyter`. No requirements.txt needed — user already has these in environment.

---

## Verification

1. Run `python src/cv.py` (add `if __name__ == "__main__":` smoke-test block) — should complete without error and print per-fold scores for LR-L1
2. Run `jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb` — all cells should execute cleanly
3. Run `jupyter nbconvert --to notebook --execute notebooks/02_models.ipynb` — model comparison table should appear; score for lr_l1 should be close to 3885 (from existing diagnostics_mp.py run); at least one model should score above baseline
