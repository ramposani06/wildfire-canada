# v14.7 = the 20 no-satellite features + latitude + longitude (22 features). Single LightGBM, no ensemble.
# Steps: train to 2021 -> validate 2022-24 -> threshold (65% recall) + Platt calibrator on 2022-24 -> train to 2024 -> score 2025+ once.
import os, json, warnings, joblib
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
warnings.filterwarnings("ignore")
FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive")
CSV  = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
INFO = "final_model_v14.6_nosat_info.json"
LAT, LON, TARGET, EPS = "LATITUDE", "LONGITUDE", "is_big_fire", 1e-4
P = dict(n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=31, is_unbalance=True, random_state=42, verbose=-1)
def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files: return os.path.join(root, name)
    raise FileNotFoundError(name)
base = json.load(open(find(INFO)))["features"]; feats = base + [LAT, LON]
df = pd.read_csv(find(CSV), low_memory=False)
df = df[df[LAT].between(41, 84) & df[LON].between(-142, -52)].dropna(subset=[TARGET]).reset_index(drop=True)
y = df[TARGET].astype(int).values; yr = df["year"].values
tr21, va, tr24, te = yr <= 2021, (yr >= 2022) & (yr <= 2024), yr <= 2024, yr >= 2025
print(f"features {len(feats)} | train<=2021 {tr21.sum():,} | validation {va.sum():,} | test 2025+ {te.sum():,} ({y[te].mean():.3f} big)")

m21 = LGBMClassifier(**P).fit(df.loc[tr21, feats], y[tr21]); sv = m21.predict_proba(df.loc[va, feats])[:, 1]; yv = y[va]
thr = float(np.sort(sv[yv == 1])[int(0.35 * (yv == 1).sum())])
pv = np.clip(sv, EPS, 1 - EPS); lr = LogisticRegression(C=1e6).fit(np.log(pv / (1 - pv)).reshape(-1, 1), yv)
cal = {"coef": float(lr.coef_[0][0]), "intercept": float(lr.intercept_[0])}
print(f"VALIDATION 2022-24: ROC {roc_auc_score(yv, sv):.3f} | PR {average_precision_score(yv, sv):.3f} | threshold {thr:.3f} | calibrator {cal}")

m24 = LGBMClassifier(**P).fit(df.loc[tr24, feats], y[tr24]); s = m24.predict_proba(df.loc[te, feats])[:, 1]; yt = y[te]
pc = np.clip(s, EPS, 1 - EPS); c = 1 / (1 + np.exp(-(cal["coef"] * np.log(pc / (1 - pc)) + cal["intercept"])))
r = np.random.RandomState(1); a, p = [], []
for _ in range(500):
    i = r.randint(0, len(yt), len(yt))
    if yt[i].sum(): a.append(roc_auc_score(yt[i], s[i])); p.append(average_precision_score(yt[i], s[i]))
ra, pa = np.percentile(a, [2.5, 97.5]), np.percentile(p, [2.5, 97.5])
print(f"\nTEST 2025+: ROC {roc_auc_score(yt, s):.3f} ({ra[0]:.3f}-{ra[1]:.3f}) | PR {average_precision_score(yt, s):.3f} ({pa[0]:.3f}-{pa[1]:.3f})")
for nm, mk in (("2025", yr[te] == 2025), ("2026", yr[te] == 2026)):
    print(f"  {nm}: n={mk.sum():,} ROC {roc_auc_score(yt[mk], s[mk]):.3f} | PR {average_precision_score(yt[mk], s[mk]):.3f}")
fl = s >= thr; tp = (fl & (yt == 1)).sum()
print(f"Alert rule (>= {thr:.3f}): flags {fl.mean():.0%}, catches {tp/yt.sum():.0%}, precision {tp/fl.sum():.0%}")
o = np.argsort(-s)
print("Top-k capture:", {f"{int(q*100)}%": f"{yt[o[:int(q*len(s))]].sum()/yt.sum():.0%}" for q in (.05, .10, .15, .20, .25)})
print(f"Calibration: Brier raw-score {brier_score_loss(yt, s):.3f} -> calibrated {brier_score_loss(yt, c):.3f} | avg chance {c.mean():.3f} vs actual {yt.mean():.3f}")
print(f"Chance at alert threshold: {1/(1+np.exp(-(cal['coef']*np.log(thr/(1-thr))+cal['intercept']))):.0%}")

final = LGBMClassifier(**P).fit(df[feats], y)
joblib.dump(final, os.path.join(FOLDER, "final_model_v14.7_allyears.pkl"))
json.dump({"features": feats, "threshold_recall": thr, "calibrator": cal, "note": "v14.7: 20 no-satellite features + latitude/longitude"},
          open(os.path.join(FOLDER, "final_model_v14.7_info.json"), "w"), indent=1)
print("\nSaved final_model_v14.7_allyears.pkl and final_model_v14.7_info.json to Drive.\nDone. Paste all of this output back.")
