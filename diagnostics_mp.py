import numpy as np, json
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from scipy.stats import ks_2samp

rng = np.random.RandomState(0)
U = "./data/"
Xtr = np.loadtxt(U+"x_train.txt", skiprows=1)
ytr = np.loadtxt(U+"y_train.txt", skiprows=1).astype(int)
Xte = np.loadtxt(U+"x_test.txt", skiprows=1)
print("shapes", Xtr.shape, ytr.shape, Xte.shape, "train pos rate", ytr.mean())

# ---- competition score helpers ----
def sweep_cutoff(y_true, p, n_vars, cap=1000):
    """Find the N (# contacted) that maximises Score = TP*10 - FP*5 - n_vars*200."""
    order = np.argsort(-p)
    best_s, best_n = -np.inf, 0
    for n in range(1, min(cap, len(p)) + 1):
        idx = order[:n]
        tp = int(np.sum(y_true[idx] == 1))
        s = tp * 10 - (n - tp) * 5 - n_vars * 200
        if s > best_s:
            best_s, best_n = s, n
    thr_at_best = float(p[order[best_n - 1]]) if best_n > 0 else 0.0
    idx_best = order[:best_n]
    tp_best = int(np.sum(y_true[idx_best] == 1))
    return best_s, best_n, thr_at_best, tp_best, best_n - tp_best

def score_at_thr(y_true, p, n_vars, thr=1/3, cap=1000):
    """Score using a fixed probability threshold (for comparison)."""
    order = np.argsort(-p)
    sel = order[[i for i, j in enumerate(order) if p[j] > thr]][:cap]
    tp = int(np.sum(y_true[sel] == 1))
    return tp * 10 - (len(sel) - tp) * 5 - n_vars * 200, tp, len(sel) - tp, len(sel)

# ---- L1 feature ranking (order of entry along C path) ----
scaler = StandardScaler().fit(Xtr)
Xs = scaler.transform(Xtr); Xtes = scaler.transform(Xte)
Cs = np.logspace(-3, 1, 25)
entry = np.full(Xtr.shape[1], np.inf)
for ci, C in enumerate(Cs):
    m = LogisticRegression(penalty='l1', solver='saga', C=C, max_iter=2000)
    m.fit(Xs, ytr)
    nz = np.where(np.abs(m.coef_[0])>1e-8)[0]
    for j in nz:
        if entry[j]==np.inf: entry[j]=ci
ranking = np.argsort(entry)
print("first 15 features entering L1 path:", (ranking[:15]+1).tolist())

# ---- sweep k, CV competition score ----
skf = StratifiedKFold(5, shuffle=True, random_state=0)
best=None
for k in range(1,21):
    feats = ranking[:k]
    p_cv = cross_val_predict(LogisticRegression(max_iter=2000),
                             Xs[:,feats], ytr, cv=skf, method='predict_proba')[:,1]
    s_opt, n_opt, thr_opt, tp_opt, fp_opt = sweep_cutoff(ytr, p_cv, k)
    s_naive, tp_naive, fp_naive, n_naive = score_at_thr(ytr, p_cv, k)
    if best is None or s_opt > best[0]: best = (s_opt, k, n_opt, thr_opt, tp_opt, fp_opt)
    print(f"k={k:2d}  opt: N={n_opt:4d} thr={thr_opt:.3f} score={s_opt:6d} TP={tp_opt} FP={fp_opt}"
          f"  | naive(1/3): score={s_naive:6d} N={n_naive} TP={tp_naive} FP={fp_naive}")
print("BEST k=%d  N=%d  thr=%.3f  score=%d  TP=%d  FP=%d" % (best[1], best[2], best[3], best[0], best[4], best[5]))

# ---- fit final calibrated model at best k, get train & test probs ----
kbest = best[1]; feats = ranking[:kbest]
clf = CalibratedClassifierCV(LogisticRegression(max_iter=2000), cv=5)
clf.fit(Xs[:,feats], ytr)
p_train = clf.predict_proba(Xs[:,feats])[:,1]
p_test  = clf.predict_proba(Xtes[:,feats])[:,1]
print("\n=== PROB DISTRIBUTIONS (best k=%d) ==="%kbest)
print("train mean %.3f median %.3f  | test mean %.3f median %.3f"%(
      p_train.mean(), np.median(p_train), p_test.mean(), np.median(p_test)))
for q in [10,25,50,75,90]:
    print(f"  p{q}: train {np.percentile(p_train,q):.3f}  test {np.percentile(p_test,q):.3f}")
ks = ks_2samp(p_train, p_test)
print("KS train vs test probs: stat %.4f p %.2e"%(ks.statistic, ks.pvalue))
print("frac p>0.5: train %.3f test %.3f"%( (p_train>.5).mean(), (p_test>.5).mean()))
print("frac p>1/3: train %.3f test %.3f"%( (p_train>1/3).mean(), (p_test>1/3).mean()))

# ---- threshold sweep on final CV probs (train) ----
p_cv_final = cross_val_predict(LogisticRegression(max_iter=2000), Xs[:,feats], ytr, cv=5, method='predict_proba')[:,1]
s_opt_f, n_opt_f, thr_opt_f, tp_opt_f, fp_opt_f = sweep_cutoff(ytr, p_cv_final, kbest)
s_naive_f, tp_naive_f, fp_naive_f, n_naive_f = score_at_thr(ytr, p_cv_final, kbest)
print("\n=== THRESHOLD SWEEP (final model, CV probs) ===")
print("optimal:     N=%d  thr=%.4f  score=%d  TP=%d  FP=%d" % (n_opt_f, thr_opt_f, s_opt_f, tp_opt_f, fp_opt_f))
print("naive (1/3): N=%d  thr=0.333  score=%d  TP=%d  FP=%d" % (n_naive_f, s_naive_f, tp_naive_f, fp_naive_f))
print("score gap (opt - naive): %d" % (s_opt_f - s_naive_f))
# show score at every 50-customer increment to reveal the curve shape
print("score curve (train CV, step=50):")
order_f = np.argsort(-p_cv_final)
for n in range(50, min(1001, len(ytr)+1), 50):
    idx = order_f[:n]; tp = int(np.sum(ytr[idx]==1)); fp = n-tp
    s = tp*10 - fp*5 - kbest*200
    print(f"  N={n:4d}  p_cut={p_cv_final[order_f[n-1]]:.3f}  score={s:6d}  TP={tp}  FP={fp}")

# ---- covariate shift: train-vs-test discriminator AUC (on selected feats + all) ----
from sklearn.model_selection import cross_val_score
def shift_auc(A,B):
    X = np.vstack([A,B]); y=np.r_[np.zeros(len(A)),np.ones(len(B))]
    return cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5, scoring='roc_auc').mean()
print("\n=== COVARIATE SHIFT (AUC: 0.5=identical, 1.0=fully separable) ===")
print("selected feats AUC: %.3f"%shift_auc(Xs[:,feats], Xtes[:,feats]))
# subsample cols for speed on full
cols = rng.choice(Xtr.shape[1], 100, replace=False)
print("100 random feats AUC: %.3f"%shift_auc(Xs[:,cols], Xtes[:,cols]))

# ---- per-feature KS on selected features ----
print("\n=== PER-FEATURE train vs test (selected) ===")
for j in feats:
    k2 = ks_2samp(Xtr[:,j], Xte[:,j])
    print(f"  V{j+1}: KS {k2.statistic:.3f} p {k2.pvalue:.2e}  trainmean {Xtr[:,j].mean():.3f} testmean {Xte[:,j].mean():.3f}")

json.dump({"best_k":int(kbest),"feats":(feats+1).tolist(),
           "optimal_N":int(n_opt_f),"optimal_threshold":round(thr_opt_f,4),
           "score_at_optimal":int(s_opt_f),"score_at_third":int(s_naive_f),
           "threshold_gap":int(s_opt_f-s_naive_f),
           "test_frac_gt_third":float((p_test>1/3).mean()),
           "test_frac_gt_half":float((p_test>.5).mean())},
          open("summary.json","w"))
