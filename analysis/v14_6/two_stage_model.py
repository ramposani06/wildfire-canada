# Two-step scoring.   Step 1 (at report time): v14.7, 22 features, no satellite.   Step 2 (evening of the report day or next morning): v14.7 + the 4 satellite columns.
# Step 2 is only valid once the report day's satellite passes are in, so it is a RE-SCORE, not a replacement for step 1.
# Honest setup: both models are frozen (trained to 2021). Alert lines are fitted on 2022-24 (65% recall), then everything is scored once on 2025+.
import os, json, warnings, joblib
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive")
CSV  = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
INFO = "final_model_v14.6_nosat_info.json"
SAT = ["modis_count_early7d", "modis_max_frp_early7d", "viirs_count_early7d", "viirs_max_frp_early7d"]
LAT, LON, TARGET, EPS = "LATITUDE", "LONGITUDE", "is_big_fire", 1e-4
P = dict(n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=31, is_unbalance=True, random_state=42, verbose=-1)
def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files: return os.path.join(root, name)
    raise FileNotFoundError(name)
f1 = json.load(open(find(INFO)))["features"] + [LAT, LON]; f2 = f1 + SAT
df = pd.read_csv(find(CSV), low_memory=False)
df = df[df[LAT].between(41, 84) & df[LON].between(-142, -52)].dropna(subset=[TARGET]).reset_index(drop=True)
y = df[TARGET].astype(int).values; yr = df["year"].values
tr, va, te = yr <= 2021, (yr >= 2022) & (yr <= 2024), yr >= 2025
def thr65(s, yy): return float(np.sort(s[yy == 1])[int(0.35 * (yy == 1).sum())])
def platt(s, yy): p = np.clip(s, EPS, 1 - EPS); lr = LogisticRegression(C=1e6).fit(np.log(p / (1 - p)).reshape(-1, 1), yy); return float(lr.coef_[0][0]), float(lr.intercept_[0])
m1 = LGBMClassifier(**P).fit(df.loc[tr, f1], y[tr]); m2 = LGBMClassifier(**P).fit(df.loc[tr, f2], y[tr])
s1, s2 = m1.predict_proba(df[f1])[:, 1], m2.predict_proba(df[f2])[:, 1]
t1, t2 = thr65(s1[va], y[va]), thr65(s2[va], y[va]); c1, c2 = platt(s1[va], y[va]), platt(s2[va], y[va])
yt = y[te]; n = te.sum(); print(f"frozen models (trained to 2021) | alert lines: step 1 {t1:.3f}, step 2 {t2:.3f} | test 2025+: {n:,} fires, {yt.sum():,} big ({yt.mean():.1%})")
def row(name, fl):
    f = fl[te]; tp = (f & (yt == 1)).sum(); return (name, f.mean(), tp / yt.sum(), tp / max(f.sum(), 1), (f & (yt == 0)).sum())
a1, a2 = s1 >= t1, s2 >= t2
rows = [row("Step 1 only (at report time)", a1), row("Step 2 only (after the day's satellite passes)", a2), row("Step 1 OR step 2 (alert at report, add late ones)", a1 | a2), row("Step 1 AND step 2 (confirmed alerts only)", a1 & a2)]
pd.set_option("display.width", 220)
R = pd.DataFrame(rows, columns=["rule", "fires flagged", "big fires caught", "precision", "false alarms"]).set_index("rule"); R["fires flagged"] = (R["fires flagged"] * 100).round(1).astype(str) + "%"
R["big fires caught"] = (R["big fires caught"] * 100).round(1).astype(str) + "%"; R["precision"] = (R["precision"] * 100).round(1).astype(str) + "%"; print("\n" + R.to_string())
print(f"\nRanking quality on 2025+:  step 1 ROC {roc_auc_score(yt, s1[te]):.3f} PR {average_precision_score(yt, s1[te]):.3f}   |   step 2 ROC {roc_auc_score(yt, s2[te]):.3f} PR {average_precision_score(yt, s2[te]):.3f}")
miss1 = (~a1[te]) & (yt == 1); print(f"\nBig fires step 1 MISSED: {miss1.sum():,}.  Step 2 catches {(miss1 & a2[te]).sum():,} of them ({(miss1 & a2[te]).sum()/miss1.sum():.0%}).")
al1 = a1[te]; print(f"Step 1 alerts: {al1.sum():,} ({(al1 & (yt==1)).sum():,} real).  Step 2 drops {(al1 & ~a2[te]).sum():,} of them ({(al1 & ~a2[te] & (yt==0)).sum():,} of those were false alarms, {(al1 & ~a2[te] & (yt==1)).sum():,} were real big fires).")
# top-k with step 2
o = np.argsort(-s2[te]); print("Step 2 top-k capture:", {f"{int(q*100)}%": f"{yt[o[:int(q*n)]].sum()/yt.sum():.0%}" for q in (.05, .10, .15, .20, .25)})
# save step-2 model trained on all years
final = LGBMClassifier(**P).fit(df[f2], y); joblib.dump(final, os.path.join(FOLDER, "final_model_v14.8_stage2_allyears.pkl"))
json.dump({"features": f2, "threshold_recall": t2, "calibrator": {"coef": c2[0], "intercept": c2[1]}, "stage1_threshold": t1, "stage1_calibrator": {"coef": c1[0], "intercept": c1[1]},
           "note": "stage 2 = v14.7 features + 4 satellite columns; valid only after the report day's satellite passes (evening or next morning)"}, open(os.path.join(FOLDER, "final_model_v14.8_stage2_info.json"), "w"), indent=1)
print("\nSaved final_model_v14.8_stage2_allyears.pkl and final_model_v14.8_stage2_info.json to Drive.\nDone. Paste all of this output back.")
