# What is MISSING in missed big fires?  Direct test: for every MISSED big fire, find its look-alike SMALL fires (nearest 10 small fires from 2012-2021 in the
# 22 model inputs). The model cannot tell them apart, so any column that still differs between the missed fire and its twins is information the model does not have.
# Control: the same comparison for ordinary small fires vs their small twins (should be ~0, shows the noise level).
# Columns tested: every numeric column in the data (satellite, cause, FWI, nearby fires, extra geography ...) plus dryness_raw.csv / aspect_wind_raw.csv if on Drive.
# Satellite columns are marked: they are measured on the report day, so they are not available at report time.
import os, json, re, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive"); CSV = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
LAT, LON, TARGET = "LATITUDE", "LONGITUDE", "is_big_fire"; K = 10
P = dict(n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=31, is_unbalance=True, random_state=42, verbose=-1)
def find(name, must=True):
    for root, _, files in os.walk(FOLDER):
        if name in files: return os.path.join(root, name)
    if must: raise FileNotFoundError(name)
feats = json.load(open(find("final_model_v14.6_nosat_info.json")))["features"] + [LAT, LON]
df = pd.read_csv(find(CSV), low_memory=False)
df = df[df[LAT].between(41, 84) & df[LON].between(-142, -52)].dropna(subset=[TARGET]).copy(); df["_id"] = df.index.astype(str)
for f, cols in [("dryness_raw.csv", None), ("aspect_wind_raw.csv", ["u", "v", "spd", "asp_sin", "asp_cos"])]:
    p = find(f, must=False)
    if p:
        r = pd.read_csv(p).drop_duplicates("_id"); r["_id"] = r["_id"].astype(str); r = r[["_id"] + (cols or [c for c in r.columns if c != "_id"])]
        df = df.merge(r, on="_id", how="left"); print(f"merged {f}: {df[r.columns[1]].notna().mean():.0%} of fires have values")
df = df.reset_index(drop=True); y = df[TARGET].astype(int).values; yr = df["year"].values; tr, va = yr <= 2021, (yr >= 2022) & (yr <= 2024)
df["rep_dayofyear"] = pd.to_datetime(df["REP_DATE"].astype(str).str[:10], errors="coerce").dt.dayofyear
if "CAUSE" in df.columns: c0 = df["CAUSE"].astype(str).str.strip().str[0]; df["cause_human"] = (c0 == "H").astype(int); df["cause_natural"] = (c0 == "N").astype(int)
m = LGBMClassifier(**P).fit(df.loc[tr, feats], y[tr]); df["score"] = m.predict_proba(df[feats])[:, 1]
thr = float(np.sort(df.loc[va & (y == 1), "score"].values)[int(0.35 * (va & (y == 1)).sum())])
bad = re.compile(r"size|_ha\b|area|out_|outdate|end|duration|big|target|score|group|orig|^year$|^_id$|unnamed|rep_date|date$|row", re.I)
cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in feats and not bad.search(c) and df[c].nunique() > 2]
sat = [c for c in cols if "early7d" in c]
med = df.loc[tr, feats].median(); sc = StandardScaler().fit(df[feats].fillna(med)); Z = sc.transform(df[feats].fillna(med))
pool = np.where((yr >= 2012) & (yr <= 2021) & (y == 0))[0]; nn = NearestNeighbors(n_neighbors=K).fit(Z[pool])
test = np.where(yr >= 2022)[0]; miss = test[(y[test] == 1) & (df["score"].values[test] < thr)]
rng = np.random.default_rng(0); ctrl = rng.choice(test[(y[test] == 0) & (df["score"].values[test] < thr)], size=min(3000, len(miss) * 3), replace=False)
print(f"alert line {thr:.3f} | missed big fires 2022-26: {len(miss):,} | control small fires (not alerted): {len(ctrl):,} | twins: {K} nearest small fires from 2012-21 each")
def paired(idx):
    nb = pool[nn.kneighbors(Z[idx], return_distance=False)]; out = {}
    for c in cols + feats:
        v = df[c].values.astype(float); x = v[idx]; tw = np.nanmedian(v[nb], axis=1); ok = ~np.isnan(x) & ~np.isnan(tw)
        if ok.sum() < 100: continue
        sd = np.nanstd(v[pool]) or np.nan
        if np.isnan(sd) or sd == 0: continue
        dlt = (x[ok] - tw[ok]); bs = [np.mean(dlt[rng.integers(0, len(dlt), len(dlt))]) / sd for _ in range(300)]
        out[c] = (np.mean(dlt) / sd, np.percentile(bs, 2.5), np.percentile(bs, 97.5), (dlt > 0).mean(), (dlt < 0).mean(), ok.sum())
    return out
A, B = paired(miss), paired(ctrl)
rows = []
for c, (e, lo, hi, up, dn, n) in A.items():
    kind = "model input (sanity: should be ~0)" if c in feats else ("satellite, report day" if c in sat else ("new fetched" if c in ["soil1","soil2","soil3","p14","p30","p60","p90","pet30","tmean30","dsr","snow30","snow90","u","v","spd","asp_sin","asp_cos"] else "other column"))
    rows.append((c, kind, e, lo, hi, 100 * up, 100 * dn, B.get(c, (np.nan,))[0], n))
R = pd.DataFrame(rows, columns=["column", "kind", "effect (std units)", "ci low", "ci high", "% missed > twins", "% missed < twins", "control effect", "n"])
R["beyond control"] = R["effect (std units)"] - R["control effect"]; R["sig"] = ((R["ci low"] > 0) | (R["ci high"] < 0)) & (R["beyond control"].abs() > 0.1)
R = R[R["column"] != "YEAR_clean"].sort_values("beyond control", key=lambda s: -s.abs())
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
print("\nHow a MISSED big fire differs from its look-alike small fires (effect in standard deviations; positive = higher in the missed big fire). Control = ordinary small fires vs their twins:")
print(R[R["kind"] != "model input (sanity: should be ~0)"].head(30).round(3).to_string(index=False))
print("\nSanity: model inputs (twins are matched on these, so effects should be small):"); print(R[R["kind"].str.startswith("model input")].head(8).round(3).to_string(index=False))
print("\nSummary of what is NOT matched by the twins and is available before or at report time (no satellite):")
print(R[(R["kind"] != "model input (sanity: should be ~0)") & (R["kind"] != "satellite, report day") & R["sig"] & (R["effect (std units)"].abs() > 0.15)].head(15).round(3).to_string(index=False))
R.to_csv(os.path.join(FOLDER, "twin_difference_results.csv"), index=False); print("\nSaved twin_difference_results.csv to Drive.\nDone. Paste all of this output back.")
