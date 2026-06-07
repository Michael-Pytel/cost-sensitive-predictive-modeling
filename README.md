# AML Project 2: Cost-Sensitive Predictive Modeling

**Course**: Advanced Machine Learning, Warsaw University of Technology (MiNI PW)  
**Team**: Mikołaj Rowicki, Michał Pytel, Katsiaryna Bokhan (Group 1)

## Problem

Select up to 1,000 customers from a 5,000-record test set who are most likely to accept a marketing offer, while minimizing the number of features used.

**Scoring formula**:
```
Score = TP × 10 − FP × 5 − NoVariables × 200
```

Each correctly targeted customer earns €10, each false positive costs €5, and every feature used costs €200 regardless of how many customers are contacted.

## Results

| Configuration | K | Score (OOF) | TP | FP | N | Precision |
|---|---|---|---|---|---|---|
| LR-L1 baseline | 5 | 3,885 | 659 | 341 | 1000 | 65.9% |
| **ExtraTrees (final)** | **6** | **5,685** | **792** | **207** | **999** | **79.3%** |

The final model achieves a **46% improvement** over the logistic regression baseline.

**Final features**: V176, V255, V191, V160, V380, V342  
**Expected test score**: ~5,400–5,600 (based on multi-seed stability analysis, mean = 5,522 ± 105)

## Project Structure

```
├── data/                         # Raw and processed datasets
│   ├── x_train.txt               # 5000×500 training features
│   ├── y_train.txt               # Binary labels
│   ├── x_test.txt                # 5000×500 test features
│   └── processed/                # MI-reduced datasets (493 features)
│
├── src/                          # Reusable modules
│   ├── data.py                   # Data loading and scaling
│   ├── scoring.py                # Competition scoring (sweep_cutoff, n_vars_used)
│   ├── features.py               # Feature ranking (L1 path, MI, covariate shift)
│   ├── models.py                 # Model factory (LR, RF, GBM, LGBM, XGB)
│   ├── cv.py                     # Competition-aware CV evaluation
│   ├── model_generator.py        # Random hyperparameter sampling for trees
│   └── feature_importance_sorter.py  # CV-based feature importance aggregation
│
├── notebooks/                    # Analysis notebooks (run in order)
│   ├── 01_eda.ipynb              # EDA: distributions, shift, separability, calibration
│   ├── 02_first_models.ipynb     # Baseline models comparison (LR, RF, GBM, etc.)
│   ├── 03_mi_analysis.ipynb      # Pairwise MI → redundant feature removal (500→493)
│   ├── 04_trees_feature_importance.ipynb  # 800 random configs → aggregated rankings
│   ├── 05_exhaustive_search.ipynb # All 120 subsets × Optuna fine-tuning
│   └── 06_stability_blend.ipynb  # Multi-seed stability, blending, V224 ablation
│
├── scripts/
│   ├── first_diagnostics.py      # Original LR-L1 baseline diagnostics
│   ├── diagnostics_trees.py      # Tree random search over rankings
│   └── pairwise_mutual_information_calculation.py  # All-pairs MI computation
│
├── submission.py                 # ★ Final model → submission files
├── report.pdf                    # LaTeX report (5 pages)
└── README.md
```

## Pipeline

The modeling pipeline proceeds in six stages:

1. **EDA** (`01_eda.ipynb`): Verified near-50/50 label balance and — critically — **no covariate shift** between train and test (discriminator AUC ≈ 0.50, KS stats uniformly small). This confirmed that OOF scores transfer reliably to the test set.

2. **Baseline models** (`02_first_models.ipynb`): Compared LR-L1, LR-L2, RF, GBM, LightGBM, XGBoost using competition-aware CV. LR-L1 with 5 features scored 3,885. Tree models used too many features implicitly.

3. **Mutual information filtering** (`03_mi_analysis.ipynb`): Computed pairwise MI for all 124,750 feature pairs. Removed 7 redundant features (MI > 1.0 with another feature), yielding 493 features.

4. **Tree feature importance** (`04_trees_feature_importance.ipynb`): Ran 200 random hyperparameter configs × 4 model families (RF, ExtraTrees, XGBoost, LightGBM). Aggregated importances identified 7 dominant candidates: V176, V255, V191, V160, V215, V380, V342.

5. **Exhaustive subset search** (`05_exhaustive_search.ipynb`): Evaluated all 120 subsets of size k=2..7 from the 7 candidates, each with 4 strong HP configs. Then Optuna fine-tuning (80 trials) on the top 5 subsets. Best: k=6 (V176, V255, V191, V160, V380, V342), score = 5,685.

6. **Stability and ablation** (`06_stability_blend.ipynb`): Validated with 5 CV seeds (mean 5,522 ± 105). Blending multiple HPs did not improve over the single best. Adding V224 (LR top feature) degraded score — confirmed it carries no complementary signal for trees.

## Generating the Submission

```bash
python submission.py --ids STUDENT1ID_STUDENT2ID_STUDENT3ID --data-dir data/
```

This will:
1. Run OOF sanity check on training data (expected score ≈ 5,685)
2. Train the final ExtraTrees model on all 5,000 training samples
3. Generate `<IDs>_obs.txt` (customer indices) and `<IDs>_vars.txt` (feature indices)

## Requirements

- Python 3.10+
- numpy, pandas, scikit-learn, scipy, matplotlib, seaborn
- xgboost, lightgbm, optuna (for notebooks 04–05)

## Key Design Decisions

- **Feature cost drives everything**: At €200 per feature, a feature must improve precision by ≥1.33pp at N=1000 to pay for itself. This ruled out the 7th feature (V215).
- **Exhaustive > nested search**: The standard top-k prefix selection explored <10% of subsets within the candidate pool. Exhaustive enumeration found a non-nested k=6 subset that beats the nested k=7 best.
- **No domain adaptation needed**: Identical train/test distributions meant OOF-calibrated thresholds transfer directly — no need for conservative adjustments.
