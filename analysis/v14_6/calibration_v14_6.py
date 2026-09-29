# Wildfire project - CALIBRATION CHECK for v14.6
# Question: can the score be shown as a real chance ("P(big fire)")?
# Plan (no peeking at the test years):
#   1. Train the same model on years up to 2021 (out-of-sample scores for 2022-2024).
#   2. Fit two calibrators on 2022-2024 (Platt = simple curve, Isotonic = flexible steps).
#      Pick one by leave-one-year-out Brier score on 2022-2024 only.
#   3. Score 2025+ (2025 and 2026 separately): Brier, log loss, calibration slope/intercept,
#      predicted vs observed rate, reliability bins - raw vs calibrated.
#   4. Save the chosen calibrator (does NOT change the model or its ranking).
# Run in Colab after mounting Drive. Paste ALL output back.

import os, json, warnings, joblib
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, average_precision_score

# ---------- CONFIG ----------
FOLDER     = os.environ.get("WF_FOLDER", "/content/drive/MyDrive")
CSV        = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v3.csv")
MODEL_FILE = "final_model_v14.6_clean.pkl"          # trained through 2024
INFO_FILE  = "final_model_v14.6_clean_info.json"
OUT_FILE   = "final_model_v14.6_calibrator.pkl"
TARGET, YEAR, LAT, LON = "is_big_fire", "year", "LATITUDE", "LONGITUDE"
VAL_YEARS  = [2022, 2023, 2024]
SAVE       = os.environ.get("WF_SAVE", "1") == "1"
# ----------------------------
warnings.filterwarnings("ignore")
EPS = 1e-4

def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError(name)

def head(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)
def logit(p): p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))

class Platt:
    def fit(self, s, y):
        self.m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(s).reshape(-1, 1), y); return self
    def predict(self, s):
        return self.m.predict_proba(logit(s).reshape(-1, 1))[:, 1]

class Iso:
    def fit(self, s, y):
        self.m = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip").fit(s, y); return self
    def predict(self, s):
        return np.clip(self.m.predict(s), EPS, 1 - EPS)

def slope_intercept(y, p):
    m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    return float(m.coef_[0][0]), float(m.intercept_[0])

def report(name, y, p):
    p = np.clip(p, EPS, 1 - EPS)
    sl, ic = slope_intercept(y, p)
    return {"version": name, "n": len(y), "observed rate": y.mean(), "average score": p.mean(),
            "Brier": brier_score_loss(y, p), "log loss": log_loss(y, p), "slope": sl, "intercept": ic}

# ---------- load ----------
df = pd.read_csv(find(CSV), low_memory=False)
df = df[df[LAT].between(41, 84) & df[LON].between(-142, -52)].dropna(subset=[TARGET]).reset_index(drop=True)
info = json.load(open(find(INFO_FILE))); feats = info["features"]
model24 = joblib.load(find(MODEL_FILE))
y_all = df[TARGET].astype(int).values; year = df[YEAR].values
print(f"Rows used: {len(df):,} | features: {len(feats)}")

# ---------- 1. out-of-sample scores for the validation years ----------
head("1. TRAIN THE SAME MODEL ON 2021-AND-EARLIER (so 2022-24 scores are honest)")
tr = year <= 2021
m21 = clone(model24).fit(df.loc[tr, feats], y_all[tr])
val = np.isin(year, VAL_YEARS)
s_val = m21.predict_proba(df.loc[val, feats])[:, 1]; y_val = y_all[val]; yr_val = year[val]
print(f"Train rows {int(tr.sum()):,} | validation rows {int(val.sum()):,} (big rate {y_val.mean():.3f})")
for yv in VAL_YEARS:
    k = yr_val == yv
    print(f"  {yv}: n={int(k.sum()):,}  big rate {y_val[k].mean():.3f}  average raw score {s_val[k].mean():.3f}")

# ---------- 2. choose calibrator by leave-one-year-out on validation years ----------
head("2. CHOOSE THE CALIBRATOR (leave-one-year-out inside 2022-2024 only)")
loyo = {"raw": [], "Platt": [], "Isotonic": []}
for yv in VAL_YEARS:
    k = yr_val == yv
    loyo["raw"].append(brier_score_loss(y_val[k], s_val[k]))
    loyo["Platt"].append(brier_score_loss(y_val[k], Platt().fit(s_val[~k], y_val[~k]).predict(s_val[k])))
    loyo["Isotonic"].append(brier_score_loss(y_val[k], Iso().fit(s_val[~k], y_val[~k]).predict(s_val[k])))
for k_, v in loyo.items():
    print(f"  {k_:9s} Brier by held-out year {[round(x, 4) for x in v]}  average {np.mean(v):.4f}")
best = "Platt" if np.mean(loyo["Platt"]) <= np.mean(loyo["Isotonic"]) else "Isotonic"
print(f"Chosen: {best} (lower average Brier; ties go to Platt, the simpler one)")
platt = Platt().fit(s_val, y_val); iso = Iso().fit(s_val, y_val)
cal = platt if best == "Platt" else iso

# ---------- 3. test on 2025+ ----------
head("3. FORWARD TEST ON 2025+ (same model as the calibrator: trained to 2021)")
te = year >= 2025; yt = y_all[te]; yrt = year[te]
s_te = m21.predict_proba(df.loc[te, feats])[:, 1]
slices = {"2025+": np.ones(len(yt), bool), "2025": yrt == 2025, "2026": yrt == 2026}
print("Random-guess Brier for reference = rate*(1-rate). Lower Brier / log loss is better.")
print("Slope 1.0 and intercept 0.0 are perfect. Slope under 1 = scores too extreme.\n")
rows = []
for sn, m in slices.items():
    for vname, sc in (("raw", s_te[m]), ("Platt", platt.predict(s_te[m])), ("Isotonic", iso.predict(s_te[m]))):
        r = report(vname, yt[m], sc); r["slice"] = sn; r["no-skill Brier"] = yt[m].mean() * (1 - yt[m].mean()); rows.append(r)
tab = pd.DataFrame(rows)[["slice", "version", "n", "observed rate", "average score", "Brier", "no-skill Brier", "log loss", "slope", "intercept"]]
print(tab.round(3).to_string(index=False))
auc_raw = roc_auc_score(yt, s_te); auc_cal = roc_auc_score(yt, cal.predict(s_te))
print(f"\nRanking check (2025+): ROC-AUC raw {auc_raw:.3f} vs calibrated {auc_cal:.3f} "
      f"| PR-AUC raw {average_precision_score(yt, s_te):.3f} vs calibrated {average_precision_score(yt, cal.predict(s_te)):.3f}")
print("(Platt keeps the ranking exactly. Isotonic can create ties, so its ranking may drop a little.)")

# ---------- 4. reliability bins ----------
head(f"4. RELIABILITY BINS on 2025+ (10 equal-size groups; chosen calibrator = {best})")
pc = cal.predict(s_te)
b = pd.qcut(pd.Series(pc).rank(method="first"), 10, labels=False)
rel = pd.DataFrame({"bin": b, "predicted": pc, "raw score": s_te, "actual": yt}).groupby("bin").agg(
    fires=("actual", "size"), raw_score=("raw score", "mean"), predicted=("predicted", "mean"), actual=("actual", "mean"))
print(rel.round(3).to_string())
gap = (rel["predicted"] - rel["actual"]).abs()
print(f"\nBiggest gap between predicted and actual in a bin: {gap.max():.3f} | average gap: {gap.mean():.3f}")

# ---------- 5. what the alert score means ----------
head("5. WHAT THE 0.770 ALERT SCORE MEANS AS A CHANCE")
thr = info.get("threshold_recall", 0.770)
print(f"Raw score {thr:.3f} -> calibrated chance {float(cal.predict(np.array([thr]))[0]):.3f}")
for q in (0.5, 0.9, 0.95, 0.99):
    print(f"  raw score at the {int(q*100)}th percentile of 2025+ fires: {np.quantile(s_te, q):.3f} -> chance {float(cal.predict(np.array([np.quantile(s_te, q)]))[0]):.3f}")

# ---------- 6. saved (2024-trained) model with the same calibrator ----------
head("6. SAME CALIBRATOR ON THE SAVED 2024-TRAINED MODEL (check only)")
s24 = model24.predict_proba(df.loc[te, feats])[:, 1]
for sn, m in slices.items():
    r0 = report("raw", yt[m], s24[m]); r1 = report(best, yt[m], cal.predict(s24[m]))
    print(f"{sn:6s} Brier raw {r0['Brier']:.3f} -> calibrated {r1['Brier']:.3f} | slope {r0['slope']:.2f} -> {r1['slope']:.2f} | "
          f"average score {r0['average score']:.3f} -> {r1['average score']:.3f} vs actual {r1['observed rate']:.3f}")
print("(The 2024-trained model has seen more years than the 2021 one, so small differences are normal.)")

# ---------- verdict ----------
head("VERDICT")
b_raw = tab[(tab.slice == "2025+") & (tab.version == "raw")].iloc[0]
b_cal = tab[(tab.slice == "2025+") & (tab.version == best)].iloc[0]
ok_brier = b_cal["Brier"] <= b_raw["Brier"] + 0.001
ok_slope = 0.8 <= b_cal["slope"] <= 1.25
by_year = [tab[(tab.slice == s_) & (tab.version == best)].iloc[0]["average score"] - tab[(tab.slice == s_) & (tab.version == best)].iloc[0]["observed rate"] for s_ in ("2025", "2026")]
ok_year = all(abs(x) <= 0.03 for x in by_year)
print(("PASS   " if ok_brier else "CHECK  ") + f"calibrated Brier is as good as or better than raw on 2025+ (within 0.001) ({b_cal['Brier']:.3f} vs {b_raw['Brier']:.3f})")
print(("PASS   " if ok_slope else "CHECK  ") + f"calibration slope between 0.8 and 1.25 ({b_cal['slope']:.2f})")
print(("PASS   " if ok_year else "CHECK  ") + f"average chance within 3 points of the actual rate in both 2025 and 2026 ({by_year[0]:+.3f}, {by_year[1]:+.3f})")
print("\nIf all PASS: you may show the calibrated chance, with a note that it is calibrated on 2022-24 and can drift in a very different season.")
print("If CHECK: keep using the ranking rules (0.770 / top 15%) and do not show percentages.")

if SAVE:
    out = os.path.join(os.path.dirname(find(MODEL_FILE)), OUT_FILE)
    joblib.dump({"method": best, "calibrator": cal, "fit_years": VAL_YEARS, "base_model_trained_to": 2021,
                 "note": "Apply to raw predict_proba scores. Does not change ranking (Platt)."}, out)
    print(f"\nSaved calibrator: {out}")
print("\nDone. Paste all of this output back.")
