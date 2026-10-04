# Which years and which provinces move the PR-AUC?  Leave-one-out views for the main model (v14.7, 22 features).
# A) By year, rolling forward: train on all earlier years (from 2004), test one year. Then drop each year from the pooled result.
# B) Remove one year from TRAINING (train 2004-2024 without it, test 2025+): does that year help or hurt?
# C) By province, 2025+: score per province, pooled score without each province, and training without each province.
# D) Province x year grid of PR-AUC with the frozen model (train<=2021, test 2022-26).
import os, json, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive")
CSV  = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
INFO = "final_model_v14.6_nosat_info.json"
LAT, LON, TARGET = "LATITUDE", "LONGITUDE", "is_big_fire"
P = dict(n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=31, is_unbalance=True, random_state=42, verbose=-1)
def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files: return os.path.join(root, name)
    raise FileNotFoundError(name)
feats = json.load(open(find(INFO)))["features"] + [LAT, LON]
df = pd.read_csv(find(CSV), low_memory=False)
df = df[df[LAT].between(41, 84) & df[LON].between(-142, -52)].dropna(subset=[TARGET]).reset_index(drop=True)
y = df[TARGET].astype(int).values; yr = df["year"].values
prov = df["resolved_province"].fillna("unknown").values if "resolved_province" in df.columns else np.array(["?"] * len(df))
def pr(yt, s): return average_precision_score(yt, s) if 0 < yt.sum() < len(yt) else np.nan
def roc(yt, s): return roc_auc_score(yt, s) if 0 < yt.sum() < len(yt) else np.nan
def fit(trm): return LGBMClassifier(**P).fit(df.loc[trm, feats], y[trm])
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)

print("=== A) Rolling forward by year (train on all earlier years from 2004) ===")
scores = pd.Series(np.nan, index=df.index); rows = []
for Y in range(2013, 2027):
    m = fit(yr < Y); te = yr == Y; scores[te] = m.predict_proba(df.loc[te, feats])[:, 1]
    rows.append((Y, te.sum(), int(y[te].sum()), y[te].mean(), roc(y[te], scores[te]), pr(y[te], scores[te]), pr(y[te], scores[te]) / y[te].mean()))
A = pd.DataFrame(rows, columns=["year", "fires", "big", "base rate", "ROC-AUC", "PR-AUC", "lift"]).set_index("year"); print(A.round(3).to_string())
pool = yr >= 2013; ps = scores.values
print(f"\nPooled 2013-2026: ROC {roc(y[pool], ps[pool]):.3f} | PR {pr(y[pool], ps[pool]):.3f}  (average of yearly PR {A['PR-AUC'].mean():.3f})")
print("Pooled PR-AUC with one year REMOVED (change vs pooled):")
b = pr(y[pool], ps[pool]); out = {Y: pr(y[pool & (yr != Y)], ps[pool & (yr != Y)]) - b for Y in range(2013, 2027)}
print(pd.Series(out).round(4).to_string().replace("\n", " | "))

print("\n=== B) Remove ONE year from training (train 2004-2024 without it), test 2025+: PR-AUC change vs full training ===")
te = yr >= 2025; sfull = fit(yr <= 2024).predict_proba(df.loc[te, feats])[:, 1]; bf = pr(y[te], sfull); print(f"full training: PR {bf:.4f} | ROC {roc(y[te], sfull):.4f}")
res = {}
for Y in range(2004, 2025):
    s = fit((yr <= 2024) & (yr != Y)).predict_proba(df.loc[te, feats])[:, 1]; res[Y] = (pr(y[te], s) - bf, roc(y[te], s) - roc(y[te], sfull))
print(pd.DataFrame(res, index=["dPR", "dROC"]).T.round(4).T.to_string())
print("(negative = the model got worse without that year, so the year helps; positive = the year was hurting)")

print("\n=== C) By province, test 2025+ (model trained 2004-2024) ===")
tp = prov[te]; rows = []
for pv in pd.Series(tp).value_counts().index:
    mk = tp == pv
    if mk.sum() < 100: continue
    others = ~mk; s_wo = fit((yr <= 2024) & (prov != pv)).predict_proba(df.loc[te, feats])[:, 1]
    rows.append((pv, mk.sum(), int(y[te][mk].sum()), y[te][mk].mean(), roc(y[te][mk], sfull[mk]), pr(y[te][mk], sfull[mk]), pr(y[te][mk], sfull[mk]) / y[te][mk].mean(),
                 pr(y[te][others], sfull[others]) - bf, pr(y[te][mk], s_wo[mk]) - pr(y[te][mk], sfull[mk]), pr(y[te][others], s_wo[others]) - pr(y[te][others], sfull[others])))
C = pd.DataFrame(rows, columns=["province", "fires", "big", "base rate", "ROC", "PR", "lift", "pooled PR if removed from test", "own PR if removed from training", "other provinces' PR if removed from training"]).set_index("province")
print(C.sort_values("big", ascending=False).round(3).to_string())
print("Pooled PR with all provinces:", round(bf, 3), "| average of province PR (weighted by big fires):", round(float(np.average(C['PR'], weights=C['big'])), 3))

print("\n=== D) PR-AUC by province and year, frozen model (train<=2021), big fires in brackets; '-' = under 15 big fires ===")
s21 = fit(yr <= 2021).predict_proba(df[feats])[:, 1]; yrs = [2022, 2023, 2024, 2025, 2026]; provs = [p for p in C.sort_values("big", ascending=False).index[:9]]
tab = []
for pv in provs:
    r = {"province": pv}
    for Y in yrs:
        mk = (prov == pv) & (yr == Y); nb = int(y[mk].sum())
        r[Y] = f"{pr(y[mk], s21[mk]):.2f} ({nb})" if nb >= 15 and nb < mk.sum() else "-"
    tab.append(r)
print(pd.DataFrame(tab).set_index("province").to_string())
print("\nDone. Paste all of this output back.")
