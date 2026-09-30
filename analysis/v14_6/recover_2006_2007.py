# Wildfire project - recover the missing 2006-2007 fires
# SAFE by design: it first proves it can rebuild the fires you already have
# (2004-2011 rows in the unified dataset), and only then rebuilds 2006-2007.
# It writes a NEW file (..._v3.csv) and never touches v2. Paste ALL output back.

import os, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
FOLDER     = os.environ.get("WF_FOLDER", "/content/drive/MyDrive")
CSV        = "unified_dataset_2004_2026_FINAL_v2.csv"
OUT_NAME   = "unified_dataset_2004_2026_FINAL_v3.csv"
HIST_FEATS = "hist_2000_2011_ALL_FEATURES_no_weather.csv"
HIST_WX    = "hist_2000_2011_weather_all.csv"
NEW_INFO   = "final_model_v14.5_clean_info.json"
TARGET, LAT, LON, ID = "is_big_fire", "LATITUDE", "LONGITUDE", "UNIQUE_ID"
RECOVER_YEARS = (2006, 2007)
MIN_MATCH = 0.99
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
    return pd.to_datetime(d["REP_DATE"], errors="coerce").dt.year

def keys(d):
    if ID in d.columns:
        return d[ID].astype(str)
    return d["REP_DATE"].astype(str) + "|" + d[LAT].round(4).astype(str) + "|" + d[LON].round(4).astype(str)

def same(a, b):
    a = pd.Series(np.asarray(a)).reset_index(drop=True); b = pd.Series(np.asarray(b)).reset_index(drop=True)
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        return float(np.isclose(a.astype(float), b.astype(float), rtol=1e-5, atol=1e-6, equal_nan=True).mean())
    an, bn = a.isna(), b.isna()
    return float(((a.astype(str) == b.astype(str)) | (an & bn)).mean())

uni = pd.read_csv(find(CSV), low_memory=False)
uni["_y"] = year_of(uni); uni["_k"] = keys(uni)
hist = pd.read_csv(find(HIST_FEATS), low_memory=False); hist["_y"] = year_of(hist); hist["_k"] = keys(hist)
wx = pd.read_csv(find(HIST_WX), low_memory=False); wx["_k"] = keys(wx)
out_dir = os.path.dirname(find(CSV))
try:
    feats = json.load(open(find(NEW_INFO)))["features"]
except Exception:
    feats = []
    print("WARNING: could not read", NEW_INFO, "- only checking target and year")
required = [c for c in dict.fromkeys(list(feats) + [TARGET, "year"]) if c in uni.columns]

wx_u = wx.drop_duplicates("_k")
wx_extra = [c for c in wx_u.columns if c not in hist.columns]
src = hist.merge(wx_u[["_k"] + wx_extra], on="_k", how="left")
s = src.drop_duplicates("_k").set_index("_k")
k = uni[uni["_y"].between(2004, 2011)].drop_duplicates("_k").set_index("_k")
idx = k.index.intersection(s.index)
k, s_k = k.loc[idx], s.loc[idx]
print(f"Self-test uses {len(idx):,} fires that exist in BOTH the unified dataset and the history files\n")

# ---- learn how each unified column is made ----
src_rules, post_rules, rates = {}, {}, {}
def add_src(col, fn):
    try: r = same(k[col], fn(s_k))
    except Exception: return
    if r > rates.get(col, -1):
        rates[col] = r; src_rules[col] = fn; post_rules.pop(col, None)

for c in uni.columns:
    if not c.startswith("_") and c in s.columns:
        add_src(c, lambda d, c=c: d[c])
add_src("year", lambda d: d["_y"]); add_src("YEAR_clean", lambda d: d["_y"])
for c in [c for c in s.columns if "size" in c.lower() and pd.api.types.is_numeric_dtype(s[c])]:
    add_src(TARGET, lambda d, c=c: (d[c] > 100).astype(int))
    add_src(TARGET, lambda d, c=c: (d[c] >= 100).astype(int))
cat_cols = [c for c in s.columns if not c.startswith("_") and not pd.api.types.is_numeric_dtype(s[c])
            and 1 < s[c].nunique() <= 40]
if "is_AB" in k.columns:
    for c in cat_cols:
        add_src("is_AB", lambda d, c=c: (d[c].astype(str) == "AB").astype(int))
if "province_encoded" in k.columns:
    best = (-1, None, None)
    for c in cat_cols:
        tmp = pd.DataFrame({"cat": s_k[c].astype(str).values, "code": k["province_encoded"].values})
        m = tmp.groupby("cat")["code"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) else np.nan)
        pur = same(tmp["code"], tmp["cat"].map(m))
        if pur > best[0]: best = (pur, c, m)
    if best[1] is not None and best[0] > rates.get("province_encoded", -1):
        add_src("province_encoded", lambda d, c=best[1], m=best[2]: d[c].astype(str).map(m))

# columns made from other columns (after the copies are built)
def build(d):
    out = pd.DataFrame(index=d.index)
    for col, fn in src_rules.items():
        out[col] = np.asarray(fn(d))
    for col, fn in post_rules.items():
        out[col] = np.asarray(fn(out))
    return out
def add_post(col, fn):
    try: r = same(k[col], fn(build(s_k)))
    except Exception: return
    if r > rates.get(col, -1):
        rates[col] = r; post_rules[col] = fn; src_rules.pop(col, None)
cnt = [c for c in ("modis_count_early7d", "viirs_count_early7d") if c in src_rules]
if "sat_zero" in k.columns and cnt:
    add_post("sat_zero", lambda o: (o[cnt].fillna(0).sum(axis=1) == 0).astype(int))
if "sensor_available" in k.columns:
    tab = k.groupby("_y")["sensor_available"].agg(lambda x: x.mode().iloc[0])
    fill = tab.to_dict()
    if 2005 in fill and 2008 in fill and fill[2005] == fill[2008]:
        for y in RECOVER_YEARS: fill[y] = fill[2005]
    add_post("sensor_available", lambda o, f=fill: o["year"].map(f))

# keep ONLY rules that reproduced the existing fires (>= MIN_MATCH); any other column stays empty for recovered rows
src_rules = {c: f for c, f in src_rules.items() if rates.get(c, -1) >= MIN_MATCH}
post_rules = {c: f for c, f in post_rules.items() if rates.get(c, -1) >= MIN_MATCH}

# ---- self-test report ----
print("Can we rebuild each column the model needs?  (share of existing fires reproduced exactly)")
bad = []
for c in required:
    r = rates.get(c, -1)
    flag = "OK " if r >= MIN_MATCH else "FAIL"
    print(f"  {flag} {c:32s} {r:.3f}" if r >= 0 else f"  FAIL {c:32s} no rule found")
    if r < MIN_MATCH: bad.append(c)
others = [c for c in uni.columns if not c.startswith("_") and c not in required]
ok_other = [c for c in others if rates.get(c, -1) >= MIN_MATCH]
no_other = [c for c in others if c not in ok_other]
print(f"\nOther columns rebuilt fine: {len(ok_other)} | left empty for recovered rows: {no_other}")

if bad:
    print("\nNOT WRITTEN. Cannot safely rebuild:", bad)
    print("History file columns:", list(hist.columns))
    print("Weather file extra columns:", wx_extra)
    print("\nPaste this output back and I will adjust the rules.")
    raise SystemExit

# ---- rebuild 2006-2007 ----
new = src[src["_y"].isin(RECOVER_YEARS) & ~src["_k"].isin(set(uni["_k"]))].drop_duplicates("_k").set_index("_k")
n0 = len(new)
new = new[new[LAT].between(41, 84) & new[LON].between(-142, -52)]
print(f"\nRecovering {len(new):,} fires (dropped {n0 - len(new)} with impossible coordinates)")
add = build(new)
for c in uni.columns:
    if c not in add.columns and not c.startswith("_"):
        add[c] = np.nan
v3 = pd.concat([uni.drop(columns=["_y", "_k"]), add[[c for c in uni.columns if not c.startswith("_")]].reset_index(drop=True)],
               ignore_index=True)

# ---- sanity checks on the recovered rows ----
print("\nRows per year in new dataset:")
yy = v3["year"].value_counts().sort_index()
print(yy.loc[[y for y in range(2004, 2013) if y in yy.index]].to_dict())
print("Big-fire rate by year:", {int(y): round(float(v), 3) for y, v in v3.groupby("year")[TARGET].mean().loc[2004:2012].items()})
chk = [c for c in feats if c in v3.columns]
r = v3[v3["year"].isin(RECOVER_YEARS)][chk].isna().mean()
kk = v3[v3["year"].isin((2005, 2008))][chk].isna().mean()
worse = [(c, round(float(r[c]), 3), round(float(kk[c]), 3)) for c in chk if r[c] - kk[c] > 0.05]
print("Model features much emptier in recovered rows than in 2005/2008:", worse or "none")

p = os.path.join(out_dir, OUT_NAME)
v3.to_csv(p, index=False)
print(f"\nSaved {len(v3):,} rows to: {p}")
print("Done. Paste all of this output back.")
