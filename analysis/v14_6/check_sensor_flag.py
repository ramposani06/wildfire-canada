# Wildfire project - does the sensor flag help?  (with vs without)
# Compares the 24 v14.6 features against the same 24 + sensor_available / sat_zero.
# Two checks: validation (train <=2021 -> 2022-2024) and test (train <=2024 -> 2025+).
# Uses a bootstrap so you can see if a difference is REAL or just noise.
# Run in Colab after mounting Drive. Paste ALL output back.

import os, json, warnings, joblib
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
FOLDER   = os.environ.get("WF_FOLDER", "/content/drive/MyDrive")
CSV      = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v3.csv")
OLD_BASE = "final_model_v14_1.pkl"          # only for the hyperparameters
NEW_INFO = "final_model_v14.6_clean_info.json"
TARGET, YEAR, LAT, LON = "is_big_fire", "year", "LATITUDE", "LONGITUDE"
N_BOOT   = 300
# ----------------------------

def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError(name)

def get_model_feats(obj):
    model = obj.get("model", obj) if isinstance(obj, dict) else obj
    feats = (obj.get("features") or obj.get("feature_names")) if isinstance(obj, dict) else None
    if feats is None:
        feats = getattr(model, "feature_names_in_", None)
    if feats is None:
        feats = getattr(model, "feature_name_", None)
    return model, ([str(f) for f in feats] if feats is not None else None)

df = pd.read_csv(find(CSV), low_memory=False)
df = df[df[LAT].between(41, 84) & df[LON].between(-142, -52)].dropna(subset=[TARGET]).reset_index(drop=True)
y_all = df[TARGET].astype(int).values
year = df[YEAR].values

base_model, base_feats = get_model_feats(joblib.load(find(OLD_BASE)))
try:
    feats = json.load(open(find(NEW_INFO)))["features"]
except Exception:
    feats = base_feats
feats = [f for f in feats if f not in ("sensor_available", "sat_zero", "YEAR_clean", "has_modis_match", "n_modis_matches")]
print(f"Base model: {len(feats)} features | rows: {len(df):,}")

# ---- what do the flags look like? ----
print("\n== How the flags are coded ==")
for c in ("sensor_available", "sat_zero"):
    if c in df.columns:
        print(f"{c}: share filled {df[c].notna().mean():.0%}")
        by = df.groupby(YEAR)[c].mean().round(2)
        print("  average by year:", {int(k): float(v) for k, v in by.items()})
    else:
        print(f"{c}: column not found")
print("viirs_count_early7d average, before 2012 vs 2012+:",
      df.groupby(df[YEAR] >= 2012)["viirs_count_early7d"].mean().round(3).to_dict())

def fit_predict(cols, train_mask, eval_mask):
    m = clone(base_model).set_params(n_jobs=-1, verbose=-1)
    m.fit(df.loc[train_mask, cols], y_all[train_mask])
    return m.predict_proba(df.loc[eval_mask, cols])[:, 1]

def boot(y, pa, pb, seed=0):
    rng = np.random.default_rng(seed); n = len(y); d = []
    for _ in range(N_BOOT):
        s = rng.integers(0, n, n)
        if y[s].min() == y[s].max(): continue
        d.append(average_precision_score(y[s], pb[s]) - average_precision_score(y[s], pa[s]))
    d = np.array(d)
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)

def block(title, variants, drop_years=()):
    keep = ~np.isin(year, drop_years) if len(drop_years) else np.ones(len(df), bool)
    va = np.isin(year, [2022, 2023, 2024]); te = year >= 2025
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    verdicts = {}
    for stage, tr_end, ev in (("VALIDATION 2022-2024", 2021, va), ("TEST 2025+", 2024, te)):
        tr = (year <= tr_end) & keep
        y = y_all[ev]
        preds = {name: fit_predict(cols, tr, ev) for name, cols in variants.items()}
        names = list(variants); ref = preds[names[0]]
        print(f"\n{stage}  (train rows {int(tr.sum()):,}, eval rows {int(ev.sum()):,})")
        print(f"{'variant':28s} {'PR-AUC':>7s} {'AUC':>7s} | change vs first row (95% range)")
        for n in names:
            pr, au = average_precision_score(y, preds[n]), roc_auc_score(y, preds[n])
            if n == names[0]:
                print(f"{n:28s} {pr:>7.4f} {au:>7.4f} |")
            else:
                mu, lo, hi = boot(y, ref, preds[n])
                real = "REAL" if (lo > 0 or hi < 0) else "noise"
                print(f"{n:28s} {pr:>7.4f} {au:>7.4f} | {mu:+.4f}  ({lo:+.4f} to {hi:+.4f})  {real}")
                verdicts.setdefault(n, []).append((mu, lo, hi))
    return verdicts

flags = [c for c in ("sensor_available", "sat_zero") if c in df.columns]
v1 = {"24 features (v14.6)": feats}
if "sensor_available" in flags:
    v1["+ sensor_available"] = feats + ["sensor_available"]
r1 = block("BLOCK 1 - all rows (adds sensor_available only; sat_zero is empty for recovered 2006-07)", v1)

v2 = {"24 features": feats}
for c in flags: v2[f"+ {c}"] = feats + [c]
if len(flags) == 2: v2["+ both"] = feats + flags
r2 = block("BLOCK 2 - same rows without 2006-07 (so both flags are fully filled)", v2, drop_years=(2006, 2007))

print("\n" + "=" * 70); print("VERDICT"); print("=" * 70)
helped = []
for blk, r in (("block 1", r1), ("block 2", r2)):
    for name, res in r.items():
        if len(res) == 2 and all(lo > 0 for _, lo, _ in res):
            helped.append(f"{name} ({blk})")
if helped:
    print("These gave a REAL gain on both validation and test:", helped)
else:
    print("No flag gave a real gain on BOTH validation and test -> leave them out.")
print("\nDone. Paste all of this output back.")
