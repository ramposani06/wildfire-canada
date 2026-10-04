# Does over- or under-sampling beat the current setup (is_unbalance=True, i.e. big fires weighted up)?   Model: v14.7 (22 features).
# Setups: current | no balancing | random undersample of small fires (1:1 and 3:1, averaged over 5 draws) | random oversample of big fires x3 | SMOTE (if imbalanced-learn installs)
# Two forward splits. Ranking is what we compare (ROC-AUC, PR-AUC, gain vs current with a 95% range). Raw scores are not probabilities in any setup (calibrator handles that).
import os, json, warnings, subprocess, sys
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive")
CSV  = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
INFO = "final_model_v14.6_nosat_info.json"
LAT, LON, TARGET = "LATITUDE", "LONGITUDE", "is_big_fire"
P = dict(n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=31, random_state=42, verbose=-1)
def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files: return os.path.join(root, name)
    raise FileNotFoundError(name)
feats = json.load(open(find(INFO)))["features"] + [LAT, LON]
df = pd.read_csv(find(CSV), low_memory=False)
df = df[df[LAT].between(41, 84) & df[LON].between(-142, -52)].dropna(subset=[TARGET]).reset_index(drop=True)
y = df[TARGET].astype(int).values; yr = df["year"].values; X = df[feats]
try:
    from imblearn.over_sampling import SMOTE
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "imbalanced-learn"]); 
    try: from imblearn.over_sampling import SMOTE
    except Exception: SMOTE = None
def current(Xtr, ytr, Xte): return LGBMClassifier(is_unbalance=True, **P).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
def plain(Xtr, ytr, Xte): return LGBMClassifier(**P).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
def under(ratio):
    def f(Xtr, ytr, Xte):
        out = []
        for seed in range(5):
            r = np.random.RandomState(seed); pos = np.where(ytr == 1)[0]; neg = np.where(ytr == 0)[0]
            keep = np.r_[pos, r.choice(neg, min(len(neg), int(len(pos) * ratio)), replace=False)]
            out.append(LGBMClassifier(**P).fit(Xtr.iloc[keep], ytr[keep]).predict_proba(Xte)[:, 1])
        return np.mean(out, axis=0)
    return f
def over3(Xtr, ytr, Xte):
    pos = np.where(ytr == 1)[0]; idx = np.r_[np.arange(len(ytr)), pos, pos]
    return LGBMClassifier(**P).fit(Xtr.iloc[idx], ytr[idx]).predict_proba(Xte)[:, 1]
def smote(Xtr, ytr, Xte):
    Xf = Xtr.fillna(Xtr.median()); Xs, ys = SMOTE(sampling_strategy=0.5, random_state=0).fit_resample(Xf, ytr)
    return LGBMClassifier(**P).fit(Xs, ys).predict_proba(Xte.fillna(Xtr.median()))[:, 1]
SETUPS = [("current (big fires weighted up)", current), ("no balancing", plain), ("undersample small fires 1:1 (5 draws)", under(1)), ("undersample small fires 3:1 (5 draws)", under(3)), ("oversample big fires x3", over3)]
if SMOTE is not None: SETUPS.append(("SMOTE to 1:2", smote))
for name, trm, tem in (("train<=2021 -> test 2022-24", yr <= 2021, (yr >= 2022) & (yr <= 2024)), ("train<=2024 -> test 2025+", yr <= 2024, yr >= 2025)):
    print(f"\n######## {name} ########"); ytr, yt = y[trm], y[tem]; ref = None
    for lab, fn in SETUPS:
        s = fn(X[trm], ytr, X[tem])
        if ref is None: ref = s
        r = np.random.RandomState(1); g = []
        for _ in range(300):
            i = r.randint(0, len(yt), len(yt))
            if yt[i].sum(): g.append(average_precision_score(yt[i], s[i]) - average_precision_score(yt[i], ref[i]))
        print(f"  {lab:<40} ROC {roc_auc_score(yt, s):.3f} | PR {average_precision_score(yt, s):.3f} | gain vs current {np.mean(g):+.3f} ({np.percentile(g,2.5):+.3f} to {np.percentile(g,97.5):+.3f}) | mean raw score {s.mean():.2f}", flush=True)
print("\nDone. Paste all of this output back.")
