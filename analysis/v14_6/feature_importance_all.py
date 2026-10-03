# Feature importance for every feature the model uses (v14.7's 22) plus the 4 satellite columns for comparison.
# Model trained to 2024, scored on 2025+. Three views, because one view can mislead when columns are related:
#  1) gain      - how much each column is used inside the trees (can favour columns with many split points)
#  2) permutation - shuffle one column on the test fires; how much does the score fall? (5 repeats)
#  3) drop-one  - retrain without the column (v14.7 only); how much does the score fall? Shows what nothing else can replace.
# Then the same by group of columns.
import os, json, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive")
CSV  = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
INFO = "final_model_v14.6_nosat_info.json"
SAT  = ["modis_count_early7d", "modis_max_frp_early7d", "viirs_count_early7d", "viirs_max_frp_early7d"]
LAT, LON, TARGET = "LATITUDE", "LONGITUDE", "is_big_fire"
P = dict(n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=31, is_unbalance=True, random_state=42, verbose=-1)
def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files: return os.path.join(root, name)
    raise FileNotFoundError(name)
base = json.load(open(find(INFO)))["features"]; v147 = base + [LAT, LON]; allf = v147 + SAT
df = pd.read_csv(find(CSV), low_memory=False)
df = df[df[LAT].between(41, 84) & df[LON].between(-142, -52)].dropna(subset=[TARGET]).reset_index(drop=True)
y = df[TARGET].astype(int).values; yr = df["year"].values; tr, te = yr <= 2024, yr >= 2025
Xtr, Xte, ytr, yte = df.loc[tr], df.loc[te].reset_index(drop=True), y[tr], y[te]
def score(m, X, fs): s = m.predict_proba(X[fs])[:, 1]; return roc_auc_score(yte, s), average_precision_score(yte, s)
def group(f):
    if f in SAT: return "satellite (timing risk)"
    if f in (LAT, LON): return "location (lat/lon)"
    if f in ("dist_to_road_m", "pop_within_10km", "pop_within_25km"): return "roads & people"
    if f in ("NDVI", "elevation", "slope", "province_encoded"): return "land"
    return "weather (7 days before)"

def perm(m, fs, reps=5):
    r = np.random.RandomState(0); b = score(m, Xte, fs); out = {}
    for f in fs:
        d = []
        for _ in range(reps):
            X = Xte.copy(); X[f] = r.permutation(X[f].values); a, p = score(m, X, fs); d.append((b[0] - a, b[1] - p))
        out[f] = np.mean(d, 0), np.std([x[1] for x in d])
    return b, out

print(f"train<=2024 {tr.sum():,} | test 2025+ {te.sum():,} ({yte.mean():.3f} big)")
# ---- v14.7 (22 features)
m22 = LGBMClassifier(**P).fit(Xtr[v147], ytr); b22, p22 = perm(m22, v147)
gain = pd.Series(m22.booster_.feature_importance("gain"), index=v147); gain = gain / gain.sum()
print(f"\nv14.7 (22 features): ROC {b22[0]:.3f} | PR {b22[1]:.3f}\nDropping one column at a time (retrain)...", flush=True)
drop = {}
for f in v147:
    fs = [c for c in v147 if c != f]; a, p = score(LGBMClassifier(**P).fit(Xtr[fs], ytr), Xte, fs); drop[f] = (b22[0] - a, b22[1] - p)
t = pd.DataFrame({"group": [group(f) for f in v147], "gain_share_%": (gain * 100).round(1).values,
                  "perm_dPR": [p22[f][0][1] for f in v147], "perm_dPR_sd": [p22[f][1] for f in v147], "perm_dROC": [p22[f][0][0] for f in v147],
                  "drop_dPR": [drop[f][1] for f in v147], "drop_dROC": [drop[f][0] for f in v147]}, index=v147).sort_values("perm_dPR", ascending=False)
pd.set_option("display.width", 250); print("\n(d = how much the score FALLS; bigger = more important)"); print(t.round(4).to_string())
print("\nBy group (v14.7):")
g = t.groupby("group").agg(gain_share=("gain_share_%", "sum"), perm_dPR=("perm_dPR", "sum"), perm_dROC=("perm_dROC", "sum")).sort_values("perm_dPR", ascending=False)
print(g.round(3).to_string())
print("\nDrop a whole group at once (retrain):")
for gname in sorted(set(group(f) for f in v147)):
    fs = [f for f in v147 if group(f) != gname]; a, p = score(LGBMClassifier(**P).fit(Xtr[fs], ytr), Xte, fs)
    print(f"  without {gname:<26} ROC {a:.3f} ({a-b22[0]:+.3f}) | PR {p:.3f} ({p-b22[1]:+.3f})", flush=True)
# ---- all 26 including satellite, for reference
m26 = LGBMClassifier(**P).fit(Xtr[allf], ytr); b26, p26 = perm(m26, allf)
print(f"\nWith the 4 satellite columns added (26 features): ROC {b26[0]:.3f} | PR {b26[1]:.3f}  (satellite timing is not clean, reference only)")
s = pd.DataFrame({"group": [group(f) for f in allf], "perm_dPR": [p26[f][0][1] for f in allf], "perm_dROC": [p26[f][0][0] for f in allf]}, index=allf).sort_values("perm_dPR", ascending=False)
print(s.head(12).round(4).to_string())
print("\nDone. Paste all of this output back.")
