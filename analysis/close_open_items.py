# Wildfire project - close the 2 open items
#   Part 1: why are 2006 and 2007 missing from the unified dataset?
#   Part 2: do n_modis_matches / has_modis_match leak the fire's final size?
# Run in Colab after mounting Drive. Paste ALL output back.

import os, json, warnings, joblib
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
FOLDER     = os.environ.get("WF_FOLDER", "/content/drive/MyDrive")
CSV        = "unified_dataset_2004_2026_FINAL_v2.csv"
HIST_FEATS = "hist_2000_2011_ALL_FEATURES_no_weather.csv"
HIST_WX    = "hist_2000_2011_weather_all.csv"
OLD_BASE   = "final_model_v14_1.pkl"
NEW_INFO   = "final_model_v14.5_clean_info.json"
TARGET, LAT, LON, ID = "is_big_fire", "LATITUDE", "LONGITUDE", "UNIQUE_ID"
# ----------------------------

def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError(name)

def year_of(d):
    for c in ("year", "YEAR", "YEAR_clean"):
        if c in d.columns and d[c].notna().mean() > 0.9:
            return pd.to_numeric(d[c], errors="coerce")
    for c in ("REP_DATE", "rep_date"):
        if c in d.columns:
            return pd.to_datetime(d[c], errors="coerce").dt.year
    raise ValueError("no year column; columns are: " + str(list(d.columns)))

def keys(d):
    if ID in d.columns:
        return d[ID].astype(str)
    return d["REP_DATE"].astype(str) + "|" + d[LAT].round(4).astype(str) + "|" + d[LON].round(4).astype(str)

def auc(y, x):
    ok = x.notna()
    if ok.sum() < 50 or y[ok].nunique() < 2 or x[ok].nunique() < 2:
        return float("nan")
    return roc_auc_score(y[ok], x[ok])

uni = pd.read_csv(find(CSV), low_memory=False)
uni["_y"] = year_of(uni); uni["_k"] = keys(uni)
out_dir = os.path.dirname(find(CSV))

# =====================================================================
print("=" * 70); print("PART 1: the missing years 2006 and 2007"); print("=" * 70)
try:
    hist = pd.read_csv(find(HIST_FEATS), low_memory=False)
    wx   = pd.read_csv(find(HIST_WX), low_memory=False)
    print(f"History file: {len(hist):,} rows | weather file: {len(wx):,} rows")
    hist["_y"] = year_of(hist); hist["_k"] = keys(hist)
    wx_keys = set(keys(wx))
    h = hist[hist["_y"].between(2004, 2011)].copy()
    h["_in_unified"] = h["_k"].isin(set(uni["_k"]))
    h["_has_weather"] = h["_k"].isin(wx_keys)
    t = h.groupby("_y").agg(history_rows=("_k", "size"), in_unified=("_in_unified", "sum"),
                            has_weather=("_has_weather", "sum")).astype(int)
    t["missing_from_unified"] = t["history_rows"] - t["in_unified"]
    print("\nPer year (history file vs unified dataset):")
    print(t.to_string())

    miss = h[~h["_in_unified"]]
    print(f"\nRows in history but NOT in unified: {len(miss):,}")
    if len(miss):
        print("By year:", miss["_y"].value_counts().sort_index().to_dict())
        print(f"Of those, share that HAVE weather data: {miss['_has_weather'].mean():.1%}")
        common = [c for c in hist.columns if c in uni.columns and not c.startswith("_")
                  and pd.api.types.is_numeric_dtype(hist[c])]
        pres = h[h["_in_unified"]]
        diff = []
        for c in common:
            diff.append((c, miss[c].isna().mean(), pres[c].isna().mean()))
        diff = sorted(diff, key=lambda r: -(r[1] - r[2]))[:8]
        print("\nColumns that are much emptier in the missing rows than in kept rows:")
        print(f"{'column':28s} {'missing_rows_null':>18s} {'kept_rows_null':>15s}")
        for c, a, b in diff:
            print(f"{c:28s} {a:>18.1%} {b:>15.1%}")
        p = os.path.join(out_dir, "missing_2006_2007_rows.csv")
        miss.drop(columns=["_in_unified"]).to_csv(p, index=False)
        print("\nSaved the missing rows to:", p)
except Exception as e:
    print("PART 1 could not finish:", type(e).__name__, e)

# =====================================================================
print("\n" + "=" * 70); print("PART 2: MODIS match columns - leak check"); print("=" * 70)
u = uni.copy()
u["_era"] = np.where(u["_y"] <= 2011, "2004-2011", "2012+")
need = ["n_modis_matches", "has_modis_match", "modis_count_early7d"]
print("Columns present:", {c: (c in u.columns) for c in need})

print("\n(a) Coverage by era")
for era, g in u.groupby("_era"):
    row = {c: f"filled {g[c].notna().mean():.0%}, >0 {(g[c] > 0).mean():.0%}" for c in need if c in g.columns}
    print(f"  {era}: {row}")

leak_flags = []
if "n_modis_matches" in u.columns and "modis_count_early7d" in u.columns:
    print("\n(b) Is the match count bigger than the early-7-day count?")
    both = u.dropna(subset=["n_modis_matches", "modis_count_early7d"])
    for era, g in both.groupby("_era"):
        for big, gg in g.groupby(TARGET):
            share = (gg["n_modis_matches"] > gg["modis_count_early7d"]).mean()
            print(f"  {era} | big={int(big)} | match count > early7d count in {share:.0%} of fires "
                  f"(median match {gg['n_modis_matches'].median():.0f} vs early7d {gg['modis_count_early7d'].median():.0f})")
            if int(big) == 1 and share >= 0.5:
                leak_flags.append(f"{era}: for big fires, match count exceeds the early-7-day count in {share:.0%} of rows")

print("\n(c) How well does each column predict big fires ALONE? (ROC-AUC, higher = stronger)")
for era, g in u.groupby("_era"):
    print(f"  {era}: " + ", ".join(f"{c} {auc(g[TARGET], g[c]):.3f}" for c in need if c in g.columns))

if "ACQ_DATE" in u.columns and "REP_DATE" in u.columns:
    print("\n(d) Days between report date and matched satellite date")
    gap = (pd.to_datetime(u["ACQ_DATE"], errors="coerce") - pd.to_datetime(u["REP_DATE"], errors="coerce")).dt.days
    ok = gap.notna()
    print(f"  rows with both dates: {ok.sum():,}")
    if ok.sum():
        q = gap[ok].quantile([0.05, 0.5, 0.95]).round(1).to_dict()
        share7 = (gap[ok] > 7).mean()
        print(f"  gap quantiles (5%, 50%, 95%): {q} | share of matches AFTER day 7: {share7:.0%}")
        for big, gg in u[ok].groupby(TARGET):
            print(f"  big={int(big)}: median gap {gap[ok][gg.index].median():.1f} days, "
                  f"share after day 7 {(gap[ok][gg.index] > 7).mean():.0%}")
        if share7 >= 0.25:
            leak_flags.append(f"{share7:.0%} of matched detections are dated more than 7 days after the report")

print("\n(e) Does adding the two columns actually help? (train <=2021, validate 2022-2024)")
gain = None
try:
    d = uni[~(uni[LAT].between(41, 84) & uni[LON].between(-142, -52))].dropna(subset=[TARGET]).reset_index(drop=True)
    y = d[TARGET].astype(int).values
    base = joblib.load(find(OLD_BASE))
    feats = getattr(base, "feature_names_in_", None)
    if feats is None: feats = getattr(base, "feature_name_", None)
    try:
        info = json.load(open(find(NEW_INFO))); feats = info["features"]
    except Exception:
        pass
    feats = [str(f) for f in feats if f not in ("has_modis_match", "n_modis_matches")]
    extra = [c for c in ("has_modis_match", "n_modis_matches") if c in d.columns]
    yr = year_of(d)
    tr, va = (yr <= 2021).values, yr.isin([2022, 2023, 2024]).values
    def score(cols):
        m = clone(base).set_params(n_jobs=-1, verbose=-1).fit(d.loc[tr, cols], y[tr])
        p = m.predict_proba(d.loc[va, cols])[:, 1]
        return average_precision_score(y[va], p), roc_auc_score(y[va], p)
    a = score(feats); b = score(feats + extra)
    gain = b[0] - a[0]
    print(f"  without: PR-AUC {a[0]:.4f} | AUC {a[1]:.4f}")
    print(f"  with   : PR-AUC {b[0]:.4f} | AUC {b[1]:.4f}   (gain {gain:+.4f})")
    if gain >= 0.03:
        leak_flags.append(f"adding them boosts validation PR-AUC by {gain:+.3f}, which is suspiciously large")
except Exception as e:
    print("  ablation skipped:", type(e).__name__, e)

print("\n" + "=" * 70); print("VERDICT (Part 2)"); print("=" * 70)
if leak_flags:
    print("LEAK RISK - keep these two columns OUT of the model:")
    for f in leak_flags: print("  -", f)
else:
    print("No sign of leakage in these checks.")
if gain is not None and gain < 0.01:
    print(f"Also: they add only {gain:+.4f} PR-AUC, so there is no reason to use them either way.")
print("\nDone. Paste all of this output back.")
