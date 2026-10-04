# Every reasonable technique for imbalanced data, against the current setup (LightGBM with big fires weighted up).  Model inputs: v14.7 (22 features).
# Two forward splits (train<=2021 -> 2022-24, train<=2024 -> 2025+). A technique counts only if it beats the current setup on BOTH splits.
# Each technique runs inside its own try/except, so one failure does not stop the rest.  Optional: WF_ONLY="focal,bag" runs only names containing those words.
import os, json, time, warnings, subprocess, sys, traceback
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive")
CSV  = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
INFO = "final_model_v14.6_nosat_info.json"
ONLY = [w.strip().lower() for w in os.environ.get("WF_ONLY", "").split(",") if w.strip()]
LAT, LON, TARGET = "LATITUDE", "LONGITUDE", "is_big_fire"
BASE = dict(n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=31, random_state=42, verbose=-1)
def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files: return os.path.join(root, name)
    raise FileNotFoundError(name)
try: import imblearn
except Exception: subprocess.run([sys.executable, "-m", "pip", "install", "-q", "imbalanced-learn"])
feats = json.load(open(find(INFO)))["features"] + [LAT, LON]
df = pd.read_csv(find(CSV), low_memory=False)
df = df[df[LAT].between(41, 84) & df[LON].between(-142, -52)].dropna(subset=[TARGET]).reset_index(drop=True)
y = df[TARGET].astype(int).values; yr = df["year"].values; X = df[feats]
mon = pd.to_datetime(df["REP_DATE"].astype(str).str[:10], errors="coerce").dt.month.fillna(0).astype(int).values
prov_idx = feats.index("province_encoded")

# ---------------- techniques: each takes (Xtr, ytr, Xte, ctx) and returns a score for Xte (higher = more likely big)
def lgbm(**kw): return LGBMClassifier(**{**BASE, **kw})
def current(Xtr, ytr, Xte, c): return lgbm(is_unbalance=True).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
def spw(w):
    return lambda Xtr, ytr, Xte, c: lgbm(scale_pos_weight=w).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
def focal(gamma, alpha):
    def f(Xtr, ytr, Xte, c):
        p0 = ytr.mean(); init = np.log(p0 / (1 - p0))
        def obj(y_true, z):
            ys = np.where(y_true == 1, 1.0, -1.0); pt = np.clip(1 / (1 + np.exp(-ys * z)), 1e-7, 1 - 1e-7); q = 1 - pt; lp = np.log(pt)
            g_u = gamma * q ** gamma * pt * lp - q ** (gamma + 1)
            h_u = gamma * (-gamma * q ** gamma * pt ** 2 * lp + q ** (gamma + 1) * pt * lp + q ** (gamma + 1) * pt) + (gamma + 1) * q ** (gamma + 1) * pt
            a = np.where(y_true == 1, alpha, 1 - alpha) * 2
            return a * ys * g_u, np.maximum(a * h_u, 1e-6)
        m = LGBMClassifier(objective=obj, **BASE).fit(Xtr, ytr, init_score=np.full(len(ytr), init))
        return m.predict(Xte, raw_score=True)
    return f
def ranker(obj, global_group=False, trunc=200):
    def f(Xtr, ytr, Xte, c):
        key = np.zeros(len(ytr), int) if global_group else (c["yr_tr"] * 100 + c["mon_tr"])
        order = np.argsort(key, kind="stable"); ks = key[order]; _, counts = np.unique(ks, return_counts=True)
        kw = dict(objective=obj, n_estimators=150, learning_rate=0.05, max_depth=8, num_leaves=31, random_state=42, verbose=-1)
        if obj == "lambdarank": kw.update(lambdarank_truncation_level=trunc, label_gain=[0, 1])
        m = LGBMRanker(**kw).fit(Xtr.iloc[order], ytr[order], group=counts)
        return m.predict(Xte)
    return f
def easy_ensemble(ratio, n):
    def f(Xtr, ytr, Xte, c):
        out = []
        for seed in range(n):
            r = np.random.RandomState(seed); pos = np.where(ytr == 1)[0]; neg = np.where(ytr == 0)[0]
            keep = np.r_[pos, r.choice(neg, min(len(neg), int(len(pos) * ratio)), replace=False)]
            out.append(lgbm(random_state=seed).fit(Xtr.iloc[keep], ytr[keep]).predict_proba(Xte)[:, 1])
        return np.mean(out, axis=0)
    return f
def bagged_weighted(n):
    def f(Xtr, ytr, Xte, c):
        out = []
        for seed in range(n):
            r = np.random.RandomState(seed); idx = r.choice(len(ytr), int(0.8 * len(ytr)), replace=False)
            out.append(lgbm(is_unbalance=True, random_state=seed, subsample=0.8, subsample_freq=1, colsample_bytree=0.8).fit(Xtr.iloc[idx], ytr[idx]).predict_proba(Xte)[:, 1])
        return np.mean(out, axis=0)
    return f
def hard_neg(weight):
    def f(Xtr, ytr, Xte, c):
        oof = np.zeros(len(ytr)); folds = c["yr_tr"] % 3
        for k in range(3):
            tr, va = folds != k, folds == k
            oof[va] = lgbm(is_unbalance=True).fit(Xtr[tr], ytr[tr]).predict_proba(Xtr[va])[:, 1]
        w = np.where((ytr == 0) & (oof >= 0.5), weight, 1.0)
        return lgbm(is_unbalance=True).fit(Xtr, ytr, sample_weight=w).predict_proba(Xte)[:, 1]
    return f
def resample(kind):
    def f(Xtr, ytr, Xte, c):
        from sklearn.preprocessing import StandardScaler
        med = Xtr.median(); A = Xtr.fillna(med); B = Xte.fillna(med); sc = StandardScaler().fit(A); As, Bs = sc.transform(A), sc.transform(B)
        if kind == "borderline":
            from imblearn.over_sampling import BorderlineSMOTE; Xr, yr_ = BorderlineSMOTE(sampling_strategy=0.5, random_state=0).fit_resample(As, ytr)
        elif kind == "adasyn":
            from imblearn.over_sampling import ADASYN; Xr, yr_ = ADASYN(sampling_strategy=0.5, random_state=0).fit_resample(As, ytr)
        elif kind == "smotenc":
            from imblearn.over_sampling import SMOTENC
            A2 = A.copy(); A2["province_encoded"] = A2["province_encoded"].round().astype(int)
            Xr, yr_ = SMOTENC(categorical_features=[prov_idx], sampling_strategy=0.5, random_state=0).fit_resample(A2.values, ytr); return lgbm().fit(Xr, yr_).predict_proba(B.values)[:, 1]
        elif kind == "tomek":
            from imblearn.under_sampling import TomekLinks; Xr, yr_ = TomekLinks(n_jobs=-1).fit_resample(As, ytr)
            return lgbm(is_unbalance=True).fit(Xr, yr_).predict_proba(Bs)[:, 1]
        return lgbm().fit(Xr, yr_).predict_proba(Bs)[:, 1]
    return f
def rusboost(Xtr, ytr, Xte, c):
    from imblearn.ensemble import RUSBoostClassifier; from sklearn.tree import DecisionTreeClassifier
    med = Xtr.median(); kw = dict(n_estimators=150, learning_rate=0.5, random_state=0)
    try: m = RUSBoostClassifier(estimator=DecisionTreeClassifier(max_depth=4), **kw)
    except TypeError: m = RUSBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=4), **kw)
    return m.fit(Xtr.fillna(med), ytr).predict_proba(Xte.fillna(med))[:, 1]
def balanced_rf(Xtr, ytr, Xte, c):
    from imblearn.ensemble import BalancedRandomForestClassifier; med = Xtr.median()
    return BalancedRandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=0).fit(Xtr.fillna(med), ytr).predict_proba(Xte.fillna(med))[:, 1]
def isolation(Xtr, ytr, Xte, c):
    from sklearn.ensemble import IsolationForest; med = Xtr.median()
    return -IsolationForest(n_estimators=300, random_state=0, n_jobs=-1).fit(Xtr.fillna(med)).score_samples(Xte.fillna(med))
TECH = [("0 current: big fires weighted up (is_unbalance)", current),
        ("class weight x1 (no balancing)", spw(1)), ("class weight x3", spw(3)), ("class weight x6", spw(6)), ("class weight x10", spw(10)), ("class weight x20", spw(20)),
        ("focal loss gamma 1, alpha .5", focal(1, .5)), ("focal loss gamma 2, alpha .5", focal(2, .5)), ("focal loss gamma 2, alpha .75", focal(2, .75)), ("focal loss gamma 3, alpha .75", focal(3, .75)),
        ("rank: lambdarank by month", ranker("lambdarank")), ("rank: rank_xendcg by month", ranker("rank_xendcg")), ("rank: lambdarank one global list", ranker("lambdarank", True, 300)),
        ("EasyEnsemble 1:1 x15", easy_ensemble(1, 15)), ("EasyEnsemble 3:1 x15", easy_ensemble(3, 15)), ("bagged weighted model x10", bagged_weighted(10)),
        ("RUSBoost (undersample inside boosting)", rusboost), ("Balanced random forest", balanced_rf),
        ("hard-negative mining, weight 2", hard_neg(2)), ("hard-negative mining, weight 4", hard_neg(4)),
        ("Borderline-SMOTE", resample("borderline")), ("ADASYN", resample("adasyn")), ("SMOTE-NC (province as category)", resample("smotenc")), ("Tomek-links cleaning", resample("tomek")),
        ("Isolation forest alone (anomaly view)", isolation)]
if ONLY: TECH = [t for i, t in enumerate(TECH) if i == 0 or any(w in t[0].lower() for w in ONLY)]
SPL = [("train<=2021 -> 2022-24", yr <= 2021, (yr >= 2022) & (yr <= 2024)), ("train<=2024 -> 2025+", yr <= 2024, yr >= 2025)]
rows = []
for sname, trm, tem in SPL:
    print(f"\n######## {sname} ########", flush=True)
    Xtr, ytr, Xte, yt = X[trm].reset_index(drop=True), y[trm], X[tem].reset_index(drop=True), y[tem]
    ctx = {"yr_tr": yr[trm], "mon_tr": mon[trm]}
    r = np.random.RandomState(1); idx = [r.randint(0, len(yt), len(yt)) for _ in range(300)]
    ref = None
    for name, fn in TECH:
        t0 = time.time()
        try:
            s = np.asarray(fn(Xtr, ytr, Xte, ctx), dtype=float)
        except Exception as e:
            print(f"  {name:<46} FAILED: {str(e)[:100]}", flush=True); continue
        if ref is None: ref = s
        g = [average_precision_score(yt[i], s[i]) - average_precision_score(yt[i], ref[i]) for i in idx if yt[i].sum()]
        o = np.argsort(-s); top = yt[o[:int(.15 * len(s))]].sum() / yt.sum()
        rows.append((name, sname, roc_auc_score(yt, s), average_precision_score(yt, s), np.mean(g), np.percentile(g, 2.5), np.percentile(g, 97.5), top))
        print(f"  {name:<46} ROC {rows[-1][2]:.3f} | PR {rows[-1][3]:.3f} | gain {rows[-1][4]:+.3f} ({rows[-1][5]:+.3f} to {rows[-1][6]:+.3f}) | top15% {top:.0%} | {time.time()-t0:.0f}s", flush=True)
R = pd.DataFrame(rows, columns=["technique", "split", "roc", "pr", "gain", "lo", "hi", "top15"])
print("\n######## SUMMARY (PR-AUC gain vs current; a win needs gain above +0.003 on BOTH splits) ########")
a = R[R.split == SPL[0][0]].set_index("technique"); b = R[R.split == SPL[1][0]].set_index("technique")
S = pd.DataFrame({"PR 2022-24": a["pr"], "gain 2022-24": a["gain"], "PR 2025+": b["pr"], "gain 2025+": b["gain"]}).dropna()
S["mean gain"] = (S["gain 2022-24"] + S["gain 2025+"]) / 2; S["WIN on both"] = (S["gain 2022-24"] > 0.003) & (S["gain 2025+"] > 0.003)
pd.set_option("display.width", 220); print(S.sort_values("mean gain", ascending=False).round(3).to_string())
print("\nDone. Paste all of this output back.")
