# Export every FAILED fire (missed big fires + false alarms) and look for what they share and what is missing.
# Model v14.7 trained to 2021, alert line (65% recall) from 2022-24, looked at on 2022-26 (out-of-sample).
# Final size / cause / satellite are used only to DESCRIBE fires after the fact, never as model inputs.
import os, json, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive"); CSV = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
LAT, LON, TARGET = "LATITUDE", "LONGITUDE", "is_big_fire"
P = dict(n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=31, is_unbalance=True, random_state=42, verbose=-1)
def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files: return os.path.join(root, name)
    raise FileNotFoundError(name)
feats = json.load(open(find("final_model_v14.6_nosat_info.json")))["features"] + [LAT, LON]
df = pd.read_csv(find(CSV), low_memory=False)
df = df[df[LAT].between(41, 84) & df[LON].between(-142, -52)].dropna(subset=[TARGET]).reset_index(drop=True)
df["orig_row"] = df.index; y = df[TARGET].astype(int).values; yr = df["year"].values
tr, va = yr <= 2021, (yr >= 2022) & (yr <= 2024)
m = LGBMClassifier(**P).fit(df.loc[tr, feats], y[tr]); df["score"] = m.predict_proba(df[feats])[:, 1]
thr = float(np.sort(df.loc[va & (y == 1), "score"].values)[int(0.35 * (va & (y == 1)).sum())])
d = df[yr >= 2022].copy(); d["big"] = d[TARGET].astype(int); d["alert"] = d["score"] >= thr
d["group"] = np.select([d.big.eq(1) & d.alert, d.big.eq(1) & ~d.alert, d.big.eq(0) & d.alert], ["caught big", "MISSED big", "false alarm"], "quiet small")
d["month"] = pd.to_datetime(d["REP_DATE"].astype(str).str[:10], errors="coerce").dt.month
size = next((c for c in d.columns if c.upper() == "SIZE_HA"), None)
prov = next((c for c in ("province_filled", "resolved_province", "PROVINCE") if c in d.columns), None)
print(f"alert line {thr:.3f} | fires 2022-26 {len(d):,} | " + " | ".join(f"{k} {v:,}" for k, v in d.group.value_counts().items()))
failed = d[d.group.isin(["MISSED big", "false alarm"])]
keep = ["orig_row", "group", "score", "year", "month", prov, size, "CAUSE"] + feats + ["modis_count_early7d", "viirs_count_early7d", "viirs_max_frp_early7d"]
keep = list(dict.fromkeys(c for c in keep if c and c in d.columns))
out = os.path.join(FOLDER, "failed_fires_v14_7.csv"); failed[keep].to_csv(out, index=False); print(f"Saved {len(failed):,} failed fires to {out}")
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)

# 1) What separates MISSED big from CAUGHT big, across every numeric column (not just model inputs)
num = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c]) and c not in ("big", "orig_row", "score", "alert", TARGET, "year", "month") and d[c].nunique() > 3]
M, C = d[d.group == "MISSED big"], d[d.group == "caught big"]
rows = []
for c in num:
    a, b = M[c].dropna(), C[c].dropna()
    if len(a) < 100 or len(b) < 100: continue
    sd = pd.concat([a, b]).std()
    if not sd or np.isnan(sd): continue
    both = pd.concat([M[c], C[c]]); lab = np.r_[np.zeros(len(M)), np.ones(len(C))]; ok = both.notna().values
    rows.append((c, "model input" if c in feats else "not an input", (a.mean() - b.mean()) / sd, roc_auc_score(lab[ok], both[ok]) if ok.sum() > 50 else np.nan))
R = pd.DataFrame(rows, columns=["column", "type", "std diff (missed - caught)", "AUC missed-vs-caught"]).assign(abs_=lambda t: t["std diff (missed - caught)"].abs()).sort_values("abs_", ascending=False).drop(columns="abs_")
print("\n1) Columns where MISSED big fires differ most from CAUGHT big fires (std diff: negative = lower in missed):"); print(R.head(20).round(3).to_string(index=False))

# 2) Twins: for each failed/caught fire, how many of its 15 nearest past fires (<=2021, same 22 inputs) were big?
sc = StandardScaler().fit(df.loc[tr, feats].fillna(df.loc[tr, feats].median()))
Zref = sc.transform(df.loc[tr, feats].fillna(df.loc[tr, feats].median())); yref = y[tr]
nn = NearestNeighbors(n_neighbors=15).fit(Zref)
def twin_big_share(g): 
    idx = nn.kneighbors(sc.transform(g[feats].fillna(df.loc[tr, feats].median())), return_distance=False); return yref[idx].mean(axis=1)
print("\n2) 'Twin test': share of the 15 most similar past fires that were big (the base rate is %.0f%%)" % (100 * yref.mean()))
for grp in ["caught big", "MISSED big", "false alarm", "quiet small"]:
    g = d[d.group == grp]; g = g.sample(min(len(g), 3000), random_state=0); t = twin_big_share(g)
    print(f"  {grp:12s} n={len(g):5d}  mean twin-big share {t.mean():.0%} | at most 1 of 15 twins big: {(t <= 1/15).mean():.0%} | 5+ of 15 big: {(t >= 5/15).mean():.0%}")

# 3) Groups inside the misses
Mz = sc.transform(M[feats].fillna(df.loc[tr, feats].median())); km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(Mz); M = M.assign(cluster=km.labels_)
print("\n3) The %d missed big fires split into 4 groups (by their 22 inputs):" % len(M))
cols = [c for c in ["score", size, "dist_to_road_m", "pop_within_25km", "LATITUDE", "NDVI", "temperature_2m_max_mean", "relative_humidity_2m_mean_min", "month"] if c and c in M.columns]
P3 = M.groupby("cluster").agg(n=("score", "size"), **{c: (c, "median") for c in cols}); P3["share %"] = (100 * P3.n / len(M)).round(0)
print(P3.round(2).to_string())
if prov: print("\n   top province in each group:"); print(M.groupby("cluster")[prov].agg(lambda s: ", ".join(f"{k} {v:.0%}" for k, v in s.value_counts(normalize=True).head(3).items())).to_string())
if "CAUSE" in M.columns:
    M["cause"] = M["CAUSE"].astype(str).str.strip().str[0].map({"N": "natural", "H": "human"}).fillna("other"); print("\n   cause mix in each group:"); print(pd.crosstab(M.cluster, M.cause, normalize="index").mul(100).round(0).to_string())

# 4) What is missing: data gaps in failed fires, and how much of the missing information is only available later
print("\n4) Missing values (share of rows with no value), missed vs caught big, model inputs only:")
nanr = pd.DataFrame({"missed": M[feats].isna().mean(), "caught": C[feats].isna().mean()}); nanr = nanr[(nanr.max(axis=1) > 0.01)]
print(nanr.mul(100).round(1).to_string() if len(nanr) else "  none above 1%")
sat = [c for c in ("viirs_count_early7d", "modis_count_early7d") if c in d.columns]
if sat: print("\n   Seen by satellite on report day or before (count > 0):"); print(d.groupby("group")[sat].agg(lambda s: f"{(s.fillna(0) > 0).mean():.0%}").to_string())
if size:
    print("\n   Final size of missed vs caught big fires (ha, median / 90th percentile):"); print(d[d.big == 1].groupby("group")[size].agg(["median", lambda s: s.quantile(.9)]).round(0).to_string())
print("\nDone. Paste all of this output back.")
