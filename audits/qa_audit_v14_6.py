# Wildfire project - FINAL QA AUDIT for v14.6
# Checks the SAVED model file (not just the numbers): feature integrity, reproducibility,
# confidence ranges, province/cause/year breakdowns, alert stability, top-risk capture,
# satellite/no-satellite, sensor-era behaviour, and sanity tests on the deployed model.
# Run in Colab after mounting Drive. Paste ALL output back.

import os, json, warnings, joblib
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, average_precision_score

# ---------- CONFIG ----------
FOLDER      = os.environ.get("WF_FOLDER", "/content/drive/MyDrive")
CSV         = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v3.csv")
MODEL_FILE  = "final_model_v14.6_clean.pkl"          # trained through 2024 (the one that was tested)
INFO_FILE   = "final_model_v14.6_clean_info.json"
TARGET, YEAR, LAT, LON = "is_big_fire", "year", "LATITUDE", "LONGITUDE"
N_BOOT      = 500
RUN_LOYO    = True       # leave-one-year-out check of sensor eras (about 21 fits, several minutes)
EXPECTED    = {"2025+": (0.929, 0.634), "2025": (0.923, 0.587), "2026": (0.929, 0.656)}  # from the training run
# ----------------------------
warnings.filterwarnings("ignore")

def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError(name)

results = []
def check(name, ok, note=""):
    results.append((name, ok))
    print(("PASS   " if ok else "CHECK  ") + name + (f"   ({note})" if note else ""))

def metrics(y, p):
    if len(y) < 30 or y.min() == y.max():
        return float("nan"), float("nan")
    return roc_auc_score(y, p), average_precision_score(y, p)

def ci(y, p, fn, seed=0):
    rng = np.random.default_rng(seed); n = len(y); v = []
    for _ in range(N_BOOT):
        s = rng.integers(0, n, n)
        if y[s].min() == y[s].max(): continue
        v.append(fn(y[s], p[s]))
    return np.percentile(v, 2.5), np.percentile(v, 97.5)

df = pd.read_csv(find(CSV), low_memory=False)
df = df[df[LAT].between(41, 84) & df[LON].between(-142, -52)].dropna(subset=[TARGET]).reset_index(drop=True)
y_all = df[TARGET].astype(int).values
year = df[YEAR].values
model = joblib.load(find(MODEL_FILE))
info = json.load(open(find(INFO_FILE)))
feats = info["features"]
thr = info.get("threshold_recall")
te = year >= 2025
X = df.loc[te, feats]; y = y_all[te]; yr = year[te]
p = model.predict_proba(X)[:, 1]
slices = {"2025+": np.ones(len(y), bool), "2025": yr == 2025, "2026": yr == 2026}

# ---------------- A. feature integrity ----------------
print("=" * 70); print("A. FEATURE INTEGRITY"); print("=" * 70)
mfeats = getattr(model, "feature_name_", None)
if mfeats is None: mfeats = getattr(model, "feature_names_in_", None)
check("24 features listed in the info file", len(feats) == 24, str(len(feats)))
check("info features match the model's features (names and order)",
      mfeats is not None and [str(f) for f in mfeats] == list(feats))
check("every feature exists in the dataset", all(f in df.columns for f in feats))
check("every feature is numeric", all(pd.api.types.is_numeric_dtype(df[f]) for f in feats if f in df.columns))
bad = [f for f in feats if any(k in f.lower() for k in ("size", "is_big", "out_date", "end_date", "duration", "year", "match")) and not f.lower().startswith("sunshine")]  # sunshine_duration is weather, not fire duration
check("no size / end-date / year / match columns among the features", not bad, str(bad))

# ---------------- B. reproducibility ----------------
print("\n" + "=" * 70); print("B. REPRODUCE THE REPORTED SCORES FROM THE SAVED FILE"); print("=" * 70)
for name, m in slices.items():
    a, pr = metrics(y[m], p[m]); ea, epr = EXPECTED[name]
    check(f"{name}: ROC-AUC {a:.3f} / PR-AUC {pr:.3f}  (expected {ea} / {epr})",
          abs(a - ea) < 0.006 and abs(pr - epr) < 0.006)

# ---------------- C. bootstrap ----------------
print("\n" + "=" * 70); print(f"C. 95% CONFIDENCE RANGES ({N_BOOT} bootstrap resamples)"); print("=" * 70)
for name, m in slices.items():
    a, pr = metrics(y[m], p[m])
    la, ha = ci(y[m], p[m], roc_auc_score); lp, hp = ci(y[m], p[m], average_precision_score)
    print(f"{name:6s} n={int(m.sum()):,}  big={int(y[m].sum()):,} | ROC-AUC {a:.3f} [{la:.3f}, {ha:.3f}] | PR-AUC {pr:.3f} [{lp:.3f}, {hp:.3f}]")

# ---------------- D. province ----------------
print("\n" + "=" * 70); print("D. BY PROVINCE (2025+)"); print("=" * 70)
if "province_encoded" in df.columns:
    names = {}
    if "resolved_province" in df.columns:
        tmp = df.dropna(subset=["resolved_province", "province_encoded"])
        names = tmp.groupby("province_encoded")["resolved_province"].agg(lambda s: s.mode().iloc[0]).to_dict()
    pv = df.loc[te, "province_encoded"].values
    print(f"{'province':10s} {'n':>6s} {'big':>5s} {'big rate':>8s} {'ROC-AUC':>8s} {'PR-AUC':>7s}")
    for code in sorted(pd.unique(pv[~pd.isna(pv)])):
        m = pv == code; a, pr = metrics(y[m], p[m])
        print(f"{str(names.get(code, code)):10s} {int(m.sum()):>6,d} {int(y[m].sum()):>5d} {y[m].mean():>8.3f} {a:>8.3f} {pr:>7.3f}")
    print("(nan = too few fires or only one class; small provinces are very uncertain)")

# ---------------- E. cause ----------------
print("\n" + "=" * 70); print("E. BY CAUSE (2025+)"); print("=" * 70)
cause_cols = [c for c in df.columns if "cause" in c.lower()]
if cause_cols:
    cc = cause_cols[0]; cv = df.loc[te, cc].astype(str).values
    print("using column:", cc)
    for v in pd.Series(cv).value_counts().index[:8]:
        m = cv == v; a, pr = metrics(y[m], p[m])
        print(f"{v:14s} n={int(m.sum()):>6,d} big={int(y[m].sum()):>5d} | ROC-AUC {a:.3f} | PR-AUC {pr:.3f}")
else:
    print("No cause column in the dataset, so this cannot be checked from v3.")

# ---------------- G. alert stability ----------------
print("\n" + "=" * 70); print("G. ALERT STABILITY BY YEAR"); print("=" * 70)
print(f"{'slice':6s} | {'rule':18s} {'flagged':>8s} {'recall':>7s} {'precision':>9s}")
for name, m in slices.items():
    for label, cut in ((f"score>={thr:.3f}" if thr is not None else "score>=threshold", thr), ("top 15% (own year)", float(np.quantile(p[m], 0.85)))):
        if cut is None: continue
        pred = p[m] >= cut; big = y[m] == 1
        rec = (pred & big).sum() / max(big.sum(), 1); prec = (pred & big).sum() / max(pred.sum(), 1)
        print(f"{name:6s} | {label:18s} {pred.mean():>8.1%} {rec:>7.2f} {prec:>9.2f}")

# ---------------- H. top-risk capture ----------------
print("\n" + "=" * 70); print("H. HOW MANY BIG FIRES DO THE TOP-RISK FIRES CATCH? (2025+)"); print("=" * 70)
for k in (5, 10, 15, 20, 25):
    cut = np.quantile(p, 1 - k / 100); pred = p >= cut
    cap = (pred & (y == 1)).sum() / y.sum()
    print(f"Top {k:>2d}% of fires -> catches {cap:.0%} of big fires (lift {cap / (k / 100):.1f}x)")

# ---------------- I. satellite / weather completeness ----------------
print("\n" + "=" * 70); print("I. WITH vs WITHOUT SATELLITE, COMPLETE vs MISSING WEATHER (2025+)"); print("=" * 70)
sat = [c for c in ("modis_count_early7d", "viirs_count_early7d") if c in X.columns]
if sat:
    anysat = (X[sat].fillna(0).sum(axis=1) > 0).values
    for label, m in (("any satellite detection", anysat), ("no satellite detection", ~anysat)):
        a, pr = metrics(y[m], p[m])
        print(f"{label:24s} n={int(m.sum()):>6,d} big rate {y[m].mean():.3f} | ROC-AUC {a:.3f} | PR-AUC {pr:.3f}")
wcols = [f for f in feats if any(k in f for k in ("temperature", "precipitation", "wind", "humidity", "sunshine"))]
if wcols:
    complete = X[wcols].notna().all(axis=1).values
    for label, m in (("complete weather", complete), ("some weather missing", ~complete)):
        a, pr = metrics(y[m], p[m])
        print(f"{label:24s} n={int(m.sum()):>6,d} | ROC-AUC {a:.3f} | PR-AUC {pr:.3f}")

# ---------------- J. sensor eras (leave-one-year-out) ----------------
print("\n" + "=" * 70); print("J. SENSOR-ERA BEHAVIOUR (leave-one-year-out, NOT forward-looking)"); print("=" * 70)
if RUN_LOYO:
    rows = []
    for yy in sorted(int(v) for v in np.unique(year) if v <= 2024):
        m_te = year == yy; m_tr = ~m_te
        if y_all[m_te].min() == y_all[m_te].max(): continue
        mdl = clone(model).set_params(n_jobs=-1, verbose=-1).fit(df.loc[m_tr, feats], y_all[m_tr])
        pp = mdl.predict_proba(df.loc[m_te, feats])[:, 1]
        a, pr = metrics(y_all[m_te], pp); rows.append((yy, a, pr))
        print(f"  {yy}: ROC-AUC {a:.3f} | PR-AUC {pr:.3f}")
    r = pd.DataFrame(rows, columns=["y", "auc", "pr"])
    for label, lo, hi in (("2004-2011 (MODIS only)", 2004, 2011), ("2012-2021 (MODIS+VIIRS)", 2012, 2021), ("2022-2024", 2022, 2024)):
        s = r[(r.y >= lo) & (r.y <= hi)]
        if len(s): print(f"  average {label}: ROC-AUC {s.auc.mean():.3f} | PR-AUC {s.pr.mean():.3f}")
    old, new = r[r.y <= 2011].auc.mean(), r[r.y >= 2012].auc.mean()
    check("old (MODIS-only) years score within 0.05 AUC of newer years", abs(old - new) < 0.05, f"{old:.3f} vs {new:.3f}")

# ---------------- N. sanity tests ----------------
print("\n" + "=" * 70); print("N. SANITY TESTS ON THE SAVED MODEL (change inputs, watch the score)"); print("=" * 70)
def shifted(col, delta, floor=None):
    X2 = X.copy(); X2[col] = X2[col] + delta
    if floor is not None: X2[col] = X2[col].clip(lower=floor)
    return model.predict_proba(X2)[:, 1]
tests = [("temperature_2m_max_mean", +5, None, "hotter"), ("relative_humidity_2m_mean_mean", -15, 0, "drier air"),
         ("wind_speed_10m_max_mean", +10, None, "windier"), ("precipitation_sum_sum", -10, 0, "less rain")]
for col, d, floor, label in tests:
    if col in X.columns:
        p2 = shifted(col, d, floor); ch = (p2 - p).mean()
        check(f"{label} raises the average risk score", ch > 0, f"average change {ch:+.4f}, {(p2 > p).mean():.0%} of fires went up")
if sat:
    X2 = X.copy(); X2[[c for c in feats if "modis" in c or "viirs" in c]] = 0
    p2 = model.predict_proba(X2)[:, 1]
    ch = (p2 - p)[anysat].mean() if anysat.any() else float("nan")
    check("removing satellite detections lowers the score of fires that had them", ch < 0, f"average change {ch:+.4f} on {int(anysat.sum()):,} fires")

# ---------------- summary ----------------
print("\n" + "=" * 70); print("SUMMARY"); print("=" * 70)
fails = [n for n, ok in results if not ok]
print(f"{len(results) - len(fails)} passed, {len(fails)} to check")
for n in fails: print("  CHECK:", n)
print("\nDone. Paste all of this output back.")
