# Test: location-relative anomaly features. "Is this fire's NDVI / humidity / temperature / rain unusual for THAT place and month?"
# Normal values come ONLY from training fires (2.0 x 3.0 degree grid cell x month; fallback to the month across Canada if the cell-month has under 30 fires).
# Training rows use a leave-one-out normal (their own value is not part of their own normal). Test rows use the training-year normal.
# Caveat: normals come from days with fires (not all days), so "typical" means typical on a fire day.
# Compared on two forward splits (train<=2021 -> 2022-24, train<=2024 -> 2025+) with bootstrap ranges. Controls: +month alone (season, no anomaly).
import os, json, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive"); CSV = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
LAT, LON, TARGET = "LATITUDE", "LONGITUDE", "is_big_fire"; NB = 500; MINN = 30
P = dict(n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=31, is_unbalance=True, random_state=42, verbose=-1)
def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files: return os.path.join(root, name)
    raise FileNotFoundError(name)
feats = json.load(open(find("final_model_v14.6_nosat_info.json")))["features"] + [LAT, LON]
df = pd.read_csv(find(CSV), low_memory=False)
df = df[df[LAT].between(41, 84) & df[LON].between(-142, -52)].dropna(subset=[TARGET]).reset_index(drop=True)
df["_d"] = pd.to_datetime(df["REP_DATE"].astype(str).str[:10], errors="coerce"); df["month"] = df["_d"].dt.month.fillna(0).astype(int)
y = df[TARGET].astype(int).values; yr = df["year"].values
cell = (np.floor(df[LAT] / 2).astype(int).astype(str) + "_" + np.floor(df[LON] / 3).astype(int).astype(str)).values; ckey = pd.Series(cell + "|" + df["month"].astype(str).values)
core = ["NDVI", "relative_humidity_2m_mean_mean", "temperature_2m_max_mean", "precipitation_sum_sum"]
weather = [c for c in feats if c not in ("NDVI", "elevation", "slope", "province_encoded", "dist_to_road_m", "pop_within_10km", "pop_within_25km", LAT, LON)]
def anomalies(cols, trm):
    out = pd.DataFrame(index=df.index)
    for c in cols:
        v = df[c].astype(float); tv = v.where(trm); one = tv.notna().astype(float)
        s_cm = tv.groupby(ckey).transform("sum"); n_cm = one.groupby(ckey).transform("sum")           # cell-month, training only
        s_m = tv.groupby(df["month"]).transform("sum"); n_m = one.groupby(df["month"]).transform("sum")  # month across Canada, training only
        own = np.where(trm & v.notna(), 1.0, 0.0); vo = v.fillna(0.0) * own
        mean_cm = (s_cm - vo) / (n_cm - own).replace(0, np.nan); mean_m = (s_m - vo) / (n_m - own).replace(0, np.nan)
        use_cm = (n_cm - own) >= MINN; out[c + "_anom"] = v - np.where(use_cm, mean_cm, mean_m)
    return out
rng = np.random.default_rng(0)
def ci(d): return f"{np.mean(d):+.3f} ({np.percentile(d, 2.5):+.3f} to {np.percentile(d, 97.5):+.3f})"
def run(name, trm, tem):
    A4, AW = anomalies(core, trm), anomalies(weather, trm); X = pd.concat([df[feats], A4, AW.add_suffix("_w"), df[["month"]]], axis=1); yt = y[tem]
    sets = {"baseline (22 inputs)": feats, "+ 4 anomalies (NDVI, humidity, temp, rain)": feats + list(A4.columns), "+ month only (control: season, no anomaly)": feats + ["month"],
            "+ all 13 weather anomalies": feats + [c + "_anom_w" for c in weather], "+ NDVI anomaly only": feats + ["NDVI_anom"]}
    print(f"\n######## {name}: train {trm.sum():,} | test {tem.sum():,} fires, {yt.sum():,} big ########"); sc = {}
    for k, cols in sets.items(): sc[k] = LGBMClassifier(**P).fit(X.loc[trm, cols], y[trm]).predict_proba(X.loc[tem, cols])[:, 1]
    base = sc["baseline (22 inputs)"]; print(f"baseline  ROC {roc_auc_score(yt, base):.3f} PR {average_precision_score(yt, base):.3f}"); res = {}
    for k, s in sc.items():
        if k.startswith("baseline"): continue
        dp, dr = [], []
        for _ in range(NB):
            b = rng.integers(0, len(yt), len(yt))
            if yt[b].sum(): dp.append(average_precision_score(yt[b], s[b]) - average_precision_score(yt[b], base[b])); dr.append(roc_auc_score(yt[b], s[b]) - roc_auc_score(yt[b], base[b]))
        o = np.argsort(-s)[:int(.15 * len(s))]; o0 = np.argsort(-base)[:int(.15 * len(s))]
        print(f"  {k:46s} PR {average_precision_score(yt, s):.3f} | PR gain {ci(dp)} | ROC gain {ci(dr)} | top15% {yt[o].sum() / yt.sum():.0%} vs {yt[o0].sum() / yt.sum():.0%}"); res[k] = np.percentile(dp, 2.5)
    return res
r1 = run("train<=2021 -> 2022-24", yr <= 2021, (yr >= 2022) & (yr <= 2024)); r2 = run("train<=2024 -> 2025+", yr <= 2024, yr >= 2025)
print("\nVERDICT (kept only if the PR gain range is above zero on BOTH splits):")
for k in r1: print(f"  {k:46s} {'GAINS ON BOTH' if r1[k] > 0 and r2[k] > 0 else 'no'}")
print("\nDone. Paste all of this output back.")
