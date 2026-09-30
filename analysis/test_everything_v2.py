# Wildfire project - check v2 (shows details instead of guessing)
import os, sqlite3, joblib
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score

# ---------- CONFIG ----------
FOLDER = "/content/drive/MyDrive"
CSV    = "unified_dataset_2004_2026_FINAL_v2.csv"
MODEL  = "final_model_v14.4.pkl"
TARGET = "is_big_fire"
THRESH = 0.29
# ----------------------------

def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError(name)

df = pd.read_csv(find(CSV), low_memory=False)
print("rows:", len(df))

# ---- 1. find the real year column ----
print("\n== Year-like columns ==")
cands = [c for c in df.columns if "year" in c.lower() or "date" in c.lower()]
for c in cands:
    s = df[c]
    if "date" in c.lower():
        s = pd.to_datetime(s, errors="coerce").dt.year
    print(f"{c:25s} filled={s.notna().sum():>7,}  min={s.min()}  max={s.max()}")
    df["_y_" + c] = s
best = [c for c in cands if df["_y_" + c].max() >= 2026]
YCOL = "_y_" + (best[0] if best else cands[0])
print("using year source:", YCOL[3:])
print("\nRows per year:")
print(df[YCOL].value_counts(dropna=False).sort_index().to_string())

# ---- 2. big-fire rate ----
print("\n== Big-fire rate ==")
print("overall:", round(df[TARGET].mean(), 4), "| missing target:", df[TARGET].isna().sum())
print(df.groupby(YCOL)[TARGET].mean().round(3).to_string())

# ---- 3. coordinates ----
print("\n== Coordinates ==")
lat = next(c for c in df.columns if c.lower() in ("lat", "latitude"))
lon = next(c for c in df.columns if c.lower() in ("lon", "lng", "longitude"))
print("columns used:", lat, lon)
print("missing lat/lon:", df[lat].isna().sum(), df[lon].isna().sum())
out = df[(df[lat].notna() & df[lon].notna()) & ~(df[lat].between(41, 84) & df[lon].between(-142, -52))]
print("outside Canada box:", len(out))
if len(out):
    print(out[[lat, lon, YCOL]].head(10).to_string())
    print("lat range:", df[lat].min(), df[lat].max(), "| lon range:", df[lon].min(), df[lon].max())

# ---- 4. model score ----
print("\n== Model ==")
obj = joblib.load(find(MODEL))
model = obj.get("model", obj) if isinstance(obj, dict) else obj
feats = (obj.get("features") or obj.get("feature_names")) if isinstance(obj, dict) else None
if feats is None:
    feats = getattr(model, "feature_names_in_", None)
if feats is None:
    feats = getattr(getattr(model, "calibrated_classifiers_", [None])[0], "estimator", None)
    feats = getattr(feats, "feature_name_", None)
feats = list(feats)
print("features:", len(feats), feats)

for label, mask in [("2025+", df[YCOL] >= 2025), ("2024", df[YCOL] == 2024)]:
    t = df[mask].dropna(subset=[TARGET])
    if len(t) == 0:
        print(label, "-> NO ROWS, skipped"); continue
    p = model.predict_proba(t[feats])[:, 1]
    y = t[TARGET].values
    pred = (p >= THRESH).astype(int)
    print(f"{label}: n={len(t):,}  big-rate={y.mean():.3f}  ROC-AUC={roc_auc_score(y,p):.3f}  "
          f"PR-AUC={average_precision_score(y,p):.3f}  recall@{THRESH}={recall_score(y,pred):.2f}  "
          f"precision@{THRESH}={precision_score(y,pred):.2f}")
print("\nDone. Paste all of this output back.")
