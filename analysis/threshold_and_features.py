# Wildfire project - threshold finder + feature check for v14.4
# Run in Colab after mounting Drive.

import os, joblib
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score

# ---------- CONFIG ----------
FOLDER        = "/content/drive/MyDrive"
CSV           = "unified_dataset_2004_2026_FINAL_v2.csv"
MODEL         = "final_model_v14.4.pkl"
TARGET        = "is_big_fire"
YEAR          = "year"
VALID_YEARS   = (2022, 2023, 2024)     # used to CHOOSE the threshold
TEST_FROM     = 2025                   # used to CHECK the threshold
TARGET_RECALL = 0.65
# Features your summary says were rejected (edit if needed):
REJECTED = ["dist_to_water_m", "dist_to_settlement_m",
            "fires_25km_7d", "fires_25km_14d", "fires_50km_7d", "fires_50km_14d",
            "road_km_within_5km", "road_km_within_10km"]
# ----------------------------

def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError(name)

df = pd.read_csv(find(CSV), low_memory=False)
obj = joblib.load(find(MODEL))
model = obj.get("model", obj) if isinstance(obj, dict) else obj
feats = (obj.get("features") or obj.get("feature_names")) if isinstance(obj, dict) else None
if feats is None:
    feats = getattr(model, "feature_names_in_", None)
if feats is None:
    est = getattr(getattr(model, "calibrated_classifiers_", [None])[0], "estimator", None)
    feats = getattr(est, "feature_name_", None)
feats = [str(f) for f in feats]

# ---- 1. feature check ----
print("== Feature check ==")
present = [f for f in REJECTED if f in feats]
print("Model has", len(feats), "features")
print("Rejected features STILL in the model:", present if present else "none")
print("FWI features in model:", [f for f in feats if "fwi" in f.lower()] or "none")
print("Year-like features in model:", [f for f in feats if "year" in f.lower()])

# ---- 2. choose threshold on validation years ----
valid = df[df[YEAR].isin(VALID_YEARS)].dropna(subset=[TARGET])
test  = df[df[YEAR] >= TEST_FROM].dropna(subset=[TARGET])
pv = model.predict_proba(valid[feats])[:, 1]; yv = valid[TARGET].values
pt = model.predict_proba(test[feats])[:, 1];  yt = test[TARGET].values

def stats(p, y, t):
    pred = (p >= t).astype(int)
    return recall_score(y, pred), precision_score(y, pred, zero_division=0), pred.mean()

grid = np.round(np.arange(0.02, 0.60, 0.01), 2)
best = min(grid, key=lambda t: abs(stats(pv, yv, t)[0] - TARGET_RECALL))

print(f"\n== Threshold search (chosen on {VALID_YEARS}, checked on {TEST_FROM}+) ==")
print(f"AUC check - valid: {roc_auc_score(yv, pv):.3f} | test: {roc_auc_score(yt, pt):.3f}")
print(f"\nBest threshold for ~{int(TARGET_RECALL*100)}% recall: {best}")
r, p_, flag = stats(pv, yv, best)
print(f"  validation: recall {r:.2f}, precision {p_:.2f}, flagged {flag:.0%} of fires")
r, p_, flag = stats(pt, yt, best)
print(f"  test 2025+: recall {r:.2f}, precision {p_:.2f}, flagged {flag:.0%} of fires")

print("\nOther thresholds (test 2025+):")
print("thresh  recall  precision  flagged")
for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.29, 0.35, 0.40, 0.50]:
    r, p_, flag = stats(pt, yt, t)
    print(f"{t:>6}  {r:>6.2f}  {p_:>9.2f}  {flag:>7.0%}")
print("\nDone. Paste all of this output back.")
