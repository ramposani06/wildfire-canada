# Wildfire project - clean retrain (v14.6, on the recovered v3 dataset)
# Honest time split + feature-group selection. Run in Colab after mounting Drive.
#
# Split:  train <= 2021 | validate 2022-2024 (used to pick features + threshold)
#         test 2025+ (used ONCE at the end, never for choices)
# Then it saves a clean final model (LightGBM, no year/month leakage, no FWI).

import os, json, warnings, joblib
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, precision_score, recall_score)
warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
FOLDER   = os.environ.get("WF_FOLDER", "/content/drive/MyDrive")
CSV      = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v3.csv")
OLD_BASE = "final_model_v14_1.pkl"      # gives the starting features + hyperparameters
OLD_V144 = "final_model_v14.4.pkl"      # only used for a side-by-side score
TARGET, YEAR, LAT, LON = "is_big_fire", "year", "LATITUDE", "LONGITUDE"

TRAIN_TO      = 2021
VALID_YEARS   = (2022, 2023, 2024)
TEST_FROM     = 2025
MIN_GAIN      = 0.01      # a feature group must add at least this much validation PR-AUC (smaller gains are noise)
TARGET_RECALL = 0.65
FLAG_TOP      = 0.15      # alternative rule: flag the top 15% of fires by score

GROUPS = {
    "satellite_flags": ["sensor_available", "sat_zero"],
    "season":          ["MONTH", "DAY"],
    "alberta_flag":    ["is_AB"],
    "access_extra":    ["dist_to_water_m", "dist_to_settlement_m", "road_km_within_5km", "road_km_within_10km"],
    "nearby_fires":    ["fires_25km_7d", "fires_25km_14d", "fires_50km_7d", "fires_50km_14d"],
}
# never used: YEAR_clean (year leak); the two MODIS match columns (they leak fire size - confirmed); FWI
NEVER = ["YEAR_clean", "has_modis_match", "n_modis_matches"]
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
        cc = getattr(model, "calibrated_classifiers_", None)
        est = getattr(cc[0], "estimator", None) if cc else model
        feats = getattr(est, "feature_name_", None)
        if feats is None and hasattr(est, "booster_"):
            feats = est.booster_.feature_name()
    return model, [str(f) for f in feats]

# ---- data ----
df = pd.read_csv(find(CSV), low_memory=False)
bad = ~(df[LAT].between(41, 84) & df[LON].between(-142, -52))
print(f"Dropping {bad.sum()} rows with impossible coordinates")
df = df[~bad].dropna(subset=[TARGET]).reset_index(drop=True)
y_all = df[TARGET].astype(int).values

base_model, base_feats = get_model_feats(joblib.load(find(OLD_BASE)))
base_feats = [f for f in base_feats if f not in NEVER and "fwi" not in f.lower()]
print(f"Starting point: {type(base_model).__name__} with {len(base_feats)} features")
print("Hyperparameters copied from it:", {k: v for k, v in base_model.get_params().items()
      if k in ("n_estimators", "learning_rate", "num_leaves", "max_depth", "min_child_samples",
               "subsample", "colsample_bytree", "reg_alpha", "reg_lambda",
               "scale_pos_weight", "class_weight", "is_unbalance")})

def nan_rate(c, yrs):
    s_ = df.loc[df[YEAR].isin(yrs), c]
    return s_.isna().mean() if len(s_) else 0.0
def usable(c):
    # skip columns that were not rebuilt for the recovered 2006-2007 fires
    return c in df.columns and c not in base_feats and nan_rate(c, [2006, 2007]) - nan_rate(c, [2005, 2008]) < 0.2
for k in GROUPS:
    GROUPS[k] = [c for c in GROUPS[k] if usable(c)]
GROUPS = {k: v for k, v in GROUPS.items() if v}

tr = (df[YEAR] <= TRAIN_TO).values
va = df[YEAR].isin(VALID_YEARS).values
te = (df[YEAR] >= TEST_FROM).values
print(f"Rows - train: {tr.sum():,} | validation: {va.sum():,} | test: {te.sum():,}")

def new_model():
    return clone(base_model).set_params(n_jobs=-1, verbose=-1)

def fit_predict(feats, train_mask, pred_mask):
    m = new_model()
    m.fit(df.loc[train_mask, feats], y_all[train_mask])
    return m, m.predict_proba(df.loc[pred_mask, feats])[:, 1]

def evaluate(feats):
    _, p = fit_predict(feats, tr, va)
    return average_precision_score(y_all[va], p), roc_auc_score(y_all[va], p)

# ---- 0. which training years help? (validation years only) ----
print("\n== Training-years check (validation 2022-2024, v14.1 features) ==")
year = df[YEAR].values
variants = {
    "2012-2021 only":                          year >= 2012,
    "2004-2021 without 2006-07 (old v2)":      ~np.isin(year, [2006, 2007]),
    "2004-2021 all (with recovered 2006-07)":  np.ones(len(df), bool),
}
base_tr = year <= TRAIN_TO
vres = {}
for name, m in variants.items():
    _, pv_ = fit_predict(base_feats, base_tr & m, va)
    vres[name] = (average_precision_score(y_all[va], pv_), roc_auc_score(y_all[va], pv_))
    print(f"  {name:42s} rows {int((base_tr & m).sum()):>7,} | PR-AUC {vres[name][0]:.4f} | AUC {vres[name][1]:.4f}")
all_name = list(variants)[-1]
best_name = max(vres, key=lambda n: vres[n][0])
if vres[best_name][0] - vres[all_name][0] < 0.003:
    best_name = all_name          # within noise: prefer the most data
print("  -> training on:", best_name)
TRAIN_FILTER = variants[best_name]
tr = base_tr & TRAIN_FILTER

# ---- 1. forward selection of feature groups (validation years only) ----
print("\n== Feature-group selection (validation 2022-2024) ==")
current = list(base_feats)
best_pr, best_auc = evaluate(current)
print(f"Start (v14.1 features): PR-AUC {best_pr:.4f} | AUC {best_auc:.4f}")
remaining, adopted = dict(GROUPS), []
while remaining:
    res = {name: evaluate(current + cols) for name, cols in remaining.items()}
    print(f"\nRound {len(adopted) + 1}: add one group to {len(current)} features")
    for name, (pr, auc) in sorted(res.items(), key=lambda kv: -kv[1][0]):
        print(f"  + {name:16s} PR-AUC {pr:.4f} ({pr - best_pr:+.4f}) | AUC {auc:.4f}")
    name = max(res, key=lambda k: res[k][0])
    if res[name][0] - best_pr >= MIN_GAIN:
        current += remaining.pop(name); adopted.append(name)
        best_pr, best_auc = res[name]
        print(f"  -> ADOPT {name}")
    else:
        print(f"  -> nothing gains >= {MIN_GAIN}; stop")
        break
selected = current
print("\nAdopted groups:", adopted or "none (v14.1 features were already best)")
print(f"Final feature count: {len(selected)}")

# ---- 2. threshold from out-of-sample validation predictions ----
_, p_val = fit_predict(selected, tr, va)
prec, rec, thr = precision_recall_curve(y_all[va], p_val)
ok = np.where(rec[:-1] >= TARGET_RECALL)[0]
threshold = float(thr[ok[-1]]) if len(ok) else float(thr[0])
print(f"\nThreshold for ~{int(TARGET_RECALL * 100)}% recall (from validation): {threshold:.3f}")

# ---- 3. final model: train <= 2024, test 2025+ once ----
train_final = (df[YEAR] <= max(VALID_YEARS)).values & TRAIN_FILTER
final_model, p_test = fit_predict(selected, train_final, te)
y_te = y_all[te]

def op(p, y, t):
    pred = p >= t
    return recall_score(y, pred), precision_score(y, pred, zero_division=0), pred.mean()

print(f"\n== TEST {TEST_FROM}+ ({te.sum():,} fires, big-rate {y_te.mean():.3f}) ==")
rows = []
def add(label, p):
    rows.append((label, roc_auc_score(y_te, p), average_precision_score(y_te, p)))
add("v14.6 clean (new)", p_test)
_, p_b = fit_predict(base_feats, train_final, te); add("retrained v14.1 features", p_b)
for label, fname in (("saved v14.1", OLD_BASE), ("saved v14.4", OLD_V144)):
    try:
        m, f = get_model_feats(joblib.load(find(fname)))
        add(label, m.predict_proba(df.loc[te, f])[:, 1])
    except Exception as e:
        print("could not score", label, "-", e)
print(f"{'model':28s} {'ROC-AUC':>8s} {'PR-AUC':>8s}")
for label, a, p in rows:
    print(f"{label:28s} {a:>8.3f} {p:>8.3f}")

print("\nPer year (v14.6 clean):")
for yr in sorted(df.loc[te, YEAR].unique()):
    mk = (df.loc[te, YEAR] == yr).values
    print(f"  {int(yr)}: n={mk.sum():,}  AUC {roc_auc_score(y_te[mk], p_test[mk]):.3f}  PR-AUC {average_precision_score(y_te[mk], p_test[mk]):.3f}")

r, pr_, fl = op(p_test, y_te, threshold)
print(f"\nRule A - threshold {threshold:.3f}: recall {r:.2f}, precision {pr_:.2f}, flags {fl:.0%} of fires")
cut = float(np.quantile(p_test, 1 - FLAG_TOP))
r, pr_, fl = op(p_test, y_te, cut)
print(f"Rule B - flag top {int(FLAG_TOP * 100)}% by score (cut {cut:.3f}): recall {r:.2f}, precision {pr_:.2f}")

# ---- 4. save ----
out_dir = os.path.dirname(find(CSV))
p1 = os.path.join(out_dir, "final_model_v14.6_clean.pkl")
joblib.dump(final_model, p1)
allyears = new_model().fit(df.loc[TRAIN_FILTER, selected], y_all[TRAIN_FILTER])
p2 = os.path.join(out_dir, "final_model_v14.6_clean_allyears.pkl")
joblib.dump(allyears, p2)
with open(os.path.join(out_dir, "final_model_v14.6_clean_info.json"), "w") as fh:
    json.dump({"features": selected, "adopted_groups": adopted, "threshold_recall": threshold,
               "threshold_target_recall": TARGET_RECALL, "flag_top_fraction": FLAG_TOP,
               "trained_through": int(max(VALID_YEARS)), "test_from": TEST_FROM}, fh, indent=2)
print(f"\nSaved:\n  {p1}  (trained <= {max(VALID_YEARS)}, tested on {TEST_FROM}+)\n  {p2}  (trained on all years, for live use)")
print("\nDone. Paste all of this output back.")
