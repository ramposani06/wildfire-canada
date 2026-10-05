# Refresh the old v3 "alert rules by year" table for the main model (v14.7, 22 features) on the repaired v4 data.
# Model trained to 2024, alert line 0.695 (65% recall on 2022-24, from the model trained to 2021). Scores 2025 and 2026 separately, with bootstrap ranges.
import os, json, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive"); CSV = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
LAT, LON, TARGET = "LATITUDE", "LONGITUDE", "is_big_fire"
P = dict(n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=31, is_unbalance=True, random_state=42, verbose=-1)
def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files: return os.path.join(root, name)
    raise FileNotFoundError(name)
info = json.load(open(find("final_model_v14.7_info.json"))); feats, THR = info["features"], info["threshold_recall"]
df = pd.read_csv(find(CSV), low_memory=False)
df = df[df[LAT].between(41, 84) & df[LON].between(-142, -52)].dropna(subset=[TARGET]).reset_index(drop=True)
y = df[TARGET].astype(int).values; yr = df["year"].values
s = LGBMClassifier(**P).fit(df.loc[yr <= 2024, feats], y[yr <= 2024]).predict_proba(df[feats])[:, 1]
rng = np.random.default_rng(0)
def ci(f, yy, ss):
    v = []
    for _ in range(500):
        b = rng.integers(0, len(yy), len(yy))
        if yy[b].sum(): v.append(f(yy[b], ss[b]))
    return np.percentile(v, 2.5), np.percentile(v, 97.5)
def rules(yy, ss):
    a = ss >= THR; tp = (a & (yy == 1)).sum(); k = int(0.15 * len(ss)); o = np.argsort(-ss)[:k]; t = yy[o].sum()
    return a.mean(), tp / yy.sum(), tp / max(a.sum(), 1), t / yy.sum(), t / k
print(f"Model v14.7 trained to 2024, alert line {THR:.3f}\n")
rows = []
for name, m in [("2025", yr == 2025), ("2026", yr == 2026), ("2025+", yr >= 2025)]:
    yy, ss = y[m], s[m]; r = ci(roc_auc_score, yy, ss); p = ci(average_precision_score, yy, ss); fl, cu, pr, t15c, t15p = rules(yy, ss)
    rows.append((name, f"{m.sum():,}", f"{yy.mean():.1%}", f"{roc_auc_score(yy, ss):.3f} ({r[0]:.3f}-{r[1]:.3f})", f"{average_precision_score(yy, ss):.3f} ({p[0]:.3f}-{p[1]:.3f})",
                 f"{fl:.0%} / {cu:.0%} / {pr:.0%}", f"{t15c:.0%} / {t15p:.0%}"))
pd.set_option("display.width", 220)
print(pd.DataFrame(rows, columns=["test", "fires", "big rate", "ROC-AUC (95%)", "PR-AUC (95%)", f"{THR:.3f} rule: flagged / caught / precision", "top 15%: caught / precision"]).to_string(index=False))
print("\nDone. Paste all of this output back.")
