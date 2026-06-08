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

| Configuration | K | Score (OOF mean ± σ) | TP | FP | N | Precision |
|---|---|---|---|---|---|---|
| LR-L1 baseline | 5 | 3,885 | 659 | 341 | 1000 | 65.9% |
| ET Random Search | 7 | 5,665 | 804 | 195 | 999 | 80.5% |
| ET Optuna k=6 | 6 | 5,593 ± 19 | 785 | 213 | 998 | 78.7% |
| **SVM-RBF Optuna k=5** | **5** | **5,638 ± 38** | **772** | **227** | **999** | **77.3%** |

Scores are means over CV seeds {0, 1, 2}; TP/FP/N reported at seed 0.
The final SVM-RBF model achieves a **45% improvement** over the LR baseline.

**Final features**: V255, V191, V160, V380, V342  
**Expected test score**: ~5,638 ± 38

## Project Structure

```
├── data/
│   ├── x_train.txt                              # 5000×500 training features
│   ├── y_train.txt                              # Binary labels
│   ├── x_test.txt                               # 5000×500 test features
│   └── processed/                               # MI-reduced datasets (493 features)
│
├── src/
│   ├── data.py                                  # Data loading and scaling
│   ├── scoring.py                               # sweep_cutoff, n_vars_used
│   ├── features.py                              # Feature ranking (L1 path, MI, shift)
│   ├── models.py                                # Model factory
│   ├── cv.py                                    # Competition-aware CV evaluation
│   ├── model_generator.py                       # Random HP sampling for trees
│   └── feature_importance_sorter.py             # CV-based importance aggregation
│
├── notebooks/
│   ├── 01_eda.ipynb                             # EDA, covariate shift, calibration
│   ├── 02_first_models.ipynb                    # Baseline model comparison
│   ├── 03_mi_analysis.ipynb                     # Pairwise MI → 500→493 features
│   ├── 04_trees_feature_importance.ipynb        # 800 random configs → 7 candidates
│   ├── 05_exhaustive_search.ipynb               # 120 subsets × Optuna (ExtraTrees)
│   ├── 06_stability_blend.ipynb                 # Multi-seed stability, V224 ablation
│   ├── 07_model_comparison.ipynb                # 15-family benchmark on final features
│   └── 08_svm_subset_search.ipynb               # 120 subsets × Optuna (SVM-RBF)
│
├── notebooks/figures/                           # Figures included in the report
│
├── scripts/
│   ├── first_diagnostics.py
│   ├── diagnostics_trees.py
│   └── pairwise_mutual_information_calculation.py
│
├── best_config.json                             # ET Optuna winner (k=6)
├── svm_best_config.json                         # SVM Optuna winner (k=5) ← final
├── submission.py                                # ★ Final model → submission files
├── requirements.txt                             # Python dependencies
├── report.pdf                                   # LaTeX report (5 pages)
└── README.md
```

## Pipeline

1. **EDA** (`01_eda.ipynb`): Verified near-50/50 label balance and — critically —
   **no covariate shift** between train and test (discriminator AUC ≈ 0.50, KS stats
   uniformly small). This confirmed that OOF scores transfer reliably to the test set.

2. **Baseline models** (`02_first_models.ipynb`): Compared LR-L1, LR-L2, RF, GBM,
   LightGBM, XGBoost with competition-aware CV. LR-L1 with 5 features scored 3,885.

3. **Mutual information filtering** (`03_mi_analysis.ipynb`): Computed pairwise MI for
   all 124,750 feature pairs. Removed 7 near-duplicate features (MI > 1.0), yielding
   493 features.

4. **Tree feature importance** (`04_trees_feature_importance.ipynb`): Ran 200 random HP
   configs × 4 model families (RF, ExtraTrees, XGBoost, LightGBM). Aggregated
   importances identified 7 dominant candidates: V176, V255, V191, V160, V215, V380,
   V342.

5. **ExtraTrees exhaustive search + Optuna** (`05_exhaustive_search.ipynb`): Evaluated
   all 120 subsets (k=2..7) of the 7 candidates, multi-seed (seeds {0,1,2}). Optuna
   (80 trials, TPE) fine-tuned HP on the top 8 subsets. Best ET result: k=6
   {V176, V255, V191, V160, V380, V342}, mean score = 5,593 ± 19.

6. **Stability and ablation** (`06_stability_blend.ipynb`): Multi-seed stability check
   on held-out seeds {42, 123, 7, 17, 99}. V224 ablation confirmed it carries no
   complementary signal.

7. **Estimator family benchmark** (`07_model_comparison.ipynb`): Benchmarked 15
   classifiers on the ET-optimal 6-feature set (multi-seed, mean ± σ). Clear linear
   ceiling (~3,830); all nonlinear models exceed 4,800. SVM-RBF (untuned) reached
   5,477 ± 31 — close to tuned ET — motivating dedicated SVM tuning.

8. **SVM-RBF exhaustive search + Optuna** (`08_svm_subset_search.ipynb`): Applied the
   same pipeline to SVM-RBF using `decision_function` for speed (~5× faster than
   Platt scaling). Best subset: k=5 {V255, V191, V160, V380, V342} (drops V176),
   mean score = **5,638 ± 38** — selected as the final submission model.

## Generating the Submission

```bash
pip install -r requirements.txt
python submission.py --ids STUDENT1ID_STUDENT2ID_STUDENT3ID --data-dir data/
```

This will:
1. Run OOF sanity check on training data using `decision_function` (expected ≈ 5,638)
2. Train the final SVM-RBF model on all 5,000 training samples
3. Generate `<IDs>_obs.txt` (999 customer indices) and `<IDs>_vars.txt` (feature indices)

HP are loaded automatically from `svm_best_config.json` if present in the working
directory; hardcoded fallback values are used otherwise.

## Key Design Decisions

- **Feature cost drives everything**: At €200 per feature, a feature must improve
  precision by ≥1.33 pp at N=1000 to pay for itself. SVM wins over ET (k=6) because
  it achieves comparable performance with one fewer feature — the €200 saving outweighs
  the 13-TP deficit (130 < 200).

- **Exhaustive > nested search**: Standard top-k prefix selection explores <6% of
  subsets within the candidate pool. Exhaustive enumeration over all 120 subsets found
  non-prefix solutions for both ET and SVM.

- **Multi-seed objective prevents split overfitting**: All Optuna searches optimise the
  mean OOF score across CV seeds {0,1,2} rather than a single split. This avoids
  selecting HP configurations that exploit a favourable fold.

- **No domain adaptation needed**: Identical train/test distributions (no covariate
  shift) mean OOF-calibrated rankings transfer directly to the test set.

- **SVM drops the top tree feature**: V176 (aggregate tree rank 1) is absent from
  SVM's optimal set. The RBF kernel captures the underlying signal structure
  differently from impurity-based splits.
