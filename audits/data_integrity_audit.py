# Wildfire project - DATA / SOURCE INTEGRITY AUDIT (v3 dataset)
# Read-only. Checks the data itself: duplicates, missing-vs-zero meaning, impossible values,
# date consistency, year-to-year jumps, province labels, the recovered 2006-07 rows,
# and (if the database opens) what the NDVI table's dates look like.
# Run in Colab after mounting Drive. Paste ALL output back.

import os, json, sqlite3, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive")
CSV    = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v3.csv")
DB     = "wildfire_corrected.db"
INFO   = "final_model_v14.6_clean_info.json"
TARGET, YEAR, LAT, LON, ID = "is_big_fire", "year", "LATITUDE", "LONGITUDE", "UNIQUE_ID"
CHECK_DB = os.environ.get("WF_CHECK_DB", "1") == "1"
# ----------------------------

def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError(name)

results = []
def check(name, ok, note=""):
    results.append((name, ok))
    print(("PASS   " if ok else "CHECK  ") + name + (f"   ({note})" if note else ""))

def head(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)

df = pd.read_csv(find(CSV), low_memory=False)
try:
    feats = json.load(open(find(INFO)))["features"]
except Exception:
    feats = []
feats = [f for f in feats if f in df.columns]
print(f"Rows: {len(df):,} | model features found: {len(feats)}")
era = pd.cut(df[YEAR], [2003, 2011, 2021, 2100], labels=["2004-2011", "2012-2021", "2022+"])

# ---------- 1. duplicates ----------
head("1. DUPLICATES")
if ID in df.columns:
    d = df[ID].duplicated().sum()
    check("no duplicate UNIQUE_ID", d == 0, f"{d} duplicates")
if "REP_DATE" in df.columns:
    key = df["REP_DATE"].astype(str) + "|" + df[LAT].round(3).astype(str) + "|" + df[LON].round(3).astype(str)
    dd = key.duplicated(keep=False)
    print(f"Fires sharing the same report date and location (to ~100 m): {int(dd.sum()):,} rows in {int(key[dd].nunique()):,} groups")
    if dd.any():
        print("  by year:", df.loc[dd, YEAR].value_counts().sort_index().to_dict())
    check("under 0.5% of rows share date+location", dd.mean() < 0.005, f"{dd.mean():.2%}")
if feats:
    fd = df.duplicated(subset=feats + [TARGET], keep=False)
    check("under 0.5% of rows are exact copies on all 24 features", fd.mean() < 0.005, f"{fd.mean():.2%}")

# ---------- 2. missing vs zero ----------
head("2. MISSING vs ZERO (what does an empty or zero value mean?)")
sat = [c for c in ("modis_count_early7d", "modis_max_frp_early7d", "viirs_count_early7d", "viirs_max_frp_early7d") if c in df.columns]
for c in sat:
    g = df.groupby(era)[c].agg(empty=lambda s: s.isna().mean(), zero=lambda s: (s == 0).mean(), positive=lambda s: (s > 0).mean())
    print(f"\n{c}\n{g.round(3).to_string()}")
for cnt, frp in (("modis_count_early7d", "modis_max_frp_early7d"), ("viirs_count_early7d", "viirs_max_frp_early7d")):
    if cnt in df.columns and frp in df.columns:
        a = ((df[cnt] == 0) & (df[frp] > 0)).sum(); b = ((df[cnt] > 0) & ((df[frp] == 0) | df[frp].isna())).sum()
        check(f"{cnt}: no fires with count 0 but strength above 0", a == 0, f"{a} rows")
        check(f"{cnt}: no fires with count above 0 but strength 0/empty", b == 0, f"{b} rows")
if "viirs_count_early7d" in df.columns:
    pre = df[df[YEAR] < 2012]["viirs_count_early7d"]
    check("VIIRS is exactly 0 (not empty) before 2012", (pre == 0).all(), f"empty {pre.isna().mean():.1%}, non-zero {(pre > 0).mean():.1%}")
print("\nShare of EMPTY values by era for every model feature (only those with any empties):")
if feats:
    e = df[feats].isna().groupby(era).mean().T
    e = e[(e > 0).any(axis=1)]
    print(e.round(3).to_string() if len(e) else "  none - all 24 features are complete in every era")

# ---------- 3. impossible values ----------
head("3. IMPOSSIBLE OR PLACEHOLDER VALUES")
rng = {"relative_humidity_2m_mean_mean": (0, 100), "relative_humidity_2m_mean_min": (0, 100), "relative_humidity_2m_mean_max": (0, 100),
       "precipitation_sum_sum": (0, 3000), "precipitation_sum_mean": (0, 500), "wind_speed_10m_max_mean": (0, 200),
       "wind_speed_10m_max_max": (0, 250), "wind_gusts_10m_max_mean": (0, 300), "temperature_2m_max_mean": (-60, 50),
       "temperature_2m_max_max": (-60, 55), "temperature_2m_max_min": (-60, 50), "temperature_2m_min_mean": (-70, 40),
       "sunshine_duration_mean": (0, 90000), "slope": (0, 90), "elevation": (-100, 6000), "dist_to_road_m": (0, 1_000_000),
       "pop_within_10km": (0, 5e6), "pop_within_25km": (0, 2e7), "modis_count_early7d": (0, 1e5), "viirs_count_early7d": (0, 1e5),
       "modis_max_frp_early7d": (0, 1e5), "viirs_max_frp_early7d": (0, 1e5)}
if "NDVI" in df.columns:
    rng["NDVI"] = (-1, 1) if df["NDVI"].abs().max() <= 2 else (-10000, 10000)
print(f"{'feature':34s} {'min':>12s} {'median':>12s} {'max':>12s} {'outside range':>14s}")
for c in feats:
    s = df[c]; lo, hi = rng.get(c, (-np.inf, np.inf)); bad = int(((s < lo) | (s > hi)).sum())
    print(f"{c:34s} {s.min():>12.3f} {s.median():>12.3f} {s.max():>12.3f} {bad:>14,d}")
    if c in rng: check(f"{c} within plausible range", bad == 0, f"{bad} rows")
if "precipitation_sum_sum" in df.columns and "precipitation_sum_mean" in df.columns:
    check("total rain is not below its own daily average", (df["precipitation_sum_sum"] + 1e-6 >= df["precipitation_sum_mean"]).all())
if "temperature_2m_max_max" in df.columns:
    check("max of max temperature >= mean of max temperature", (df["temperature_2m_max_max"] + 1e-6 >= df["temperature_2m_max_mean"]).all())

# ---------- 4. dates ----------
head("4. DATE CONSISTENCY")
if "REP_DATE" in df.columns:
    raw = df["REP_DATE"].astype(str)
    rd = pd.to_datetime(df["REP_DATE"], errors="coerce")
    if rd.isna().any():
        print("Unreadable REP_DATE by year column:", df.loc[rd.isna(), YEAR].value_counts().sort_index().to_dict())
        print("Sample of the raw text that could not be read:", raw[rd.isna()].head(5).tolist())
        rd2 = pd.to_datetime(df["REP_DATE"], errors="coerce", format="mixed")
        print(f"With a flexible reader, still unreadable: {int(rd2.isna().sum())}")
        rd = rd2
    check("every report date can be read", rd.notna().all(), f"{int(rd.isna().sum())} unreadable")
    mism = (rd.dt.year != df[YEAR]).sum(); check("year column matches the report date", mism == 0, f"{mism} rows differ")
    for c, part in (("MONTH", rd.dt.month), ("DAY", rd.dt.day)):
        if c in df.columns:
            m = (part != df[c]).sum(); check(f"{c} column matches the report date", m == 0, f"{m} rows differ")
    print("Share of fires reported outside April-October, by era:", (~rd.dt.month.between(4, 10)).groupby(era).mean().round(3).to_dict())

# ---------- 5. year-to-year jumps ----------
head("5. YEAR-TO-YEAR JUMPS IN FEATURE VALUES (median by year, in units of the overall spread)")
cols = [c for c in ("temperature_2m_max_mean", "relative_humidity_2m_mean_mean", "wind_speed_10m_max_mean", "NDVI",
                    "elevation", "dist_to_road_m", "pop_within_10km", "sunshine_duration_mean") if c in df.columns]
flag = []
for c in cols:
    med = df.groupby(YEAR)[c].median(); iqr = df[c].quantile(0.75) - df[c].quantile(0.25)
    z = ((med - df[c].median()) / iqr) if iqr > 0 else med * 0
    bad = z[z.abs() > 1.0]
    print(f"{c:34s} years far from typical: {({int(k): round(float(v), 2) for k, v in bad.items()} or 'none')}")
    if len(bad): flag.append(c)
check("no feature has a year whose median is more than 1 spread from typical", not flag, str(flag))
print("(A jump is not automatically wrong - weather really differs by year - but static features like elevation should not jump.)")
for c in ("elevation", "dist_to_road_m", "pop_within_10km"):
    if c in df.columns:
        print(f"{c} yearly median:", {int(k): round(float(v), 1) for k, v in df.groupby(YEAR)[c].median().items()})

# ---------- 6. province labels ----------
head("6. PROVINCE CODE vs PROVINCE NAME")
if "province_encoded" in df.columns and "resolved_province" in df.columns:
    t = df.dropna(subset=["province_encoded", "resolved_province"])
    print(f"{'code':>5s} {'main name':>10s} {'rows':>8s} {'agreement':>10s}  other names")
    for code, g in t.groupby("province_encoded"):
        vc = g["resolved_province"].value_counts()
        print(f"{code:>5.0f} {vc.index[0]:>10s} {len(g):>8,d} {vc.iloc[0] / len(g):>10.1%}  {dict(vc.iloc[1:4]) if len(vc) > 1 else ''}")
    names = t.groupby("province_encoded")["resolved_province"].agg(lambda s: s.mode().iloc[0])
    dup = names[names.duplicated(keep=False)]
    check("each province code has its own name", dup.empty, f"codes sharing a name: {dup.to_dict()}")
    print("  (a code that shares a name with another is usually Parks Canada fires located inside a province)")

# ---------- 7. recovered 2006-07 rows ----------
head("7. RECOVERED 2006-2007 ROWS vs THE YEARS AROUND THEM")
if feats:
    a = df[df[YEAR].isin([2006, 2007])]; b = df[df[YEAR].isin([2005, 2008])]
    rows = []
    for c in feats:
        iqr = df[c].quantile(0.75) - df[c].quantile(0.25)
        diff = (a[c].median() - b[c].median()) / iqr if iqr > 0 else 0.0
        rows.append((c, a[c].median(), b[c].median(), diff))
    r = pd.DataFrame(rows, columns=["feature", "2006-07 median", "2005+2008 median", "difference/spread"])
    print(r.round(3).to_string(index=False))
    bad = r[r["difference/spread"].abs() > 0.5]
    check("recovered rows look like their neighbours (all medians within half a spread)", bad.empty, str(list(bad.feature)))

# ---------- 8. database look at NDVI ----------
head("8. NDVI DATES IN THE DATABASE (only looks; does not change anything)")
if CHECK_DB:
    try:
        con = sqlite3.connect("file:" + find(DB) + "?mode=ro", uri=True)
        tabs = [r[0] for r in con.execute("select name from sqlite_master where type='table' and name like '%ndvi%'")]
        print("NDVI-related tables:", tabs)
        for t in tabs:
            cols = [r[1] for r in con.execute(f"pragma table_info({t})")]
            print(f"\n{t}: columns = {cols}")
            print(pd.read_sql(f"select * from {t} limit 3", con).to_string(index=False))
        con.close()
    except Exception as e:
        print("Could not open the database:", type(e).__name__, e)
else:
    print("skipped (WF_CHECK_DB=0)")

head("SUMMARY")
fails = [n for n, ok in results if not ok]
print(f"{len(results) - len(fails)} passed, {len(fails)} to check")
for n in fails: print("  CHECK:", n)
print("\nDone. Paste all of this output back.")
