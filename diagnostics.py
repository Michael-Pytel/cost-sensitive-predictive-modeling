import numpy as np, json
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from scipy.stats import ks_2samp

rng = np.random.RandomState(0)
U = "/mnt/user-data/uploads/"
Xtr = np.loadtxt(U+"x_train.txt", skiprows=1)
ytr = np.loadtxt(U+"y_train.txt", skiprows=1).astype(int)
Xte = np.loadtxt(U+"x_test.txt", skiprows=1)
print("shapes", Xtr.shape, ytr.shape, Xte.shape, "train pos rate", ytr.mean())

# ---- competition score helper ----
def comp_score(y_true, p, k, cap=1000, thr=1/3):
    order = np.argsort(-p)
    sel = [i for i in order if p[i] > thr][:cap]
    tp = int(ytr_arr_eq(y_true, sel, 1)); fp = len(sel)-tp
    return tp*10 - fp*5 - k*200, tp, fp, len(sel)
def ytr_arr_eq(y, idx, val): return int(np.sum(y[idx]==val))

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
    s,tp,fp,n = comp_score(ytr, p_cv, k)
    if best is None or s>best[0]: best=(s,k,tp,fp,n)
    print(f"k={k:2d} CVscore={s:6d} TP={tp} FP={fp} contacted={n}")
print("BEST", best)

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
           "test_frac_gt_third":float((p_test>1/3).mean()),
           "test_frac_gt_half":float((p_test>.5).mean())},
          open("/home/claude/summary.json","w"))
