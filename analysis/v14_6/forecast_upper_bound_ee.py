# Same "next 3 days of weather" upper-bound test, but the weather comes from Google Earth Engine
# (ECMWF/ERA5_LAND/DAILY_AGGR) instead of Open-Meteo. ACTUAL weather, i.e. a perfect forecast.
# Whole-year mode: train on every fire of TRAIN_YEAR, test on every fire of TEST_YEAR (defaults 2024 -> 2025).
# Run in Colab after ee.Authenticate(). Resumable (results are appended to a CSV on Drive).
import os, json, time, warnings
import numpy as np, pandas as pd, ee
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")

FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive")
CSV    = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
INFO   = "final_model_v14.6_nosat_info.json"
PROJECT = os.environ.get("EE_PROJECT", "project-8d6e8459-b76f-4a38-a85")
TRAIN_YEARS = [int(x) for x in os.environ.get("WF_TRAIN_YEARS", os.environ.get("WF_TRAIN_YEAR", "2024")).split(",")]
TEST_YEARS  = [int(x) for x in os.environ.get("WF_TEST_YEARS",  os.environ.get("WF_TEST_YEAR",  "2025")).split(",")]
BATCH = 1000
OUT = os.path.join(FOLDER, "forecast_upper_bound_ee.csv")
NEW = ["next3_temp_max", "next3_wind_max", "next3_wind_mean", "next3_precip_sum", "next3_rh_min"]
LAT, LON, TARGET = "LATITUDE", "LONGITUDE", "is_big_fire"

def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files: return os.path.join(root, name)
    raise FileNotFoundError(name)

ee.Initialize(project=PROJECT)
ERA = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
last = ee.Date(ERA.aggregate_max("system:time_start")).format("YYYY-MM-dd").getInfo()
print("ERA5-Land daily data in Earth Engine runs to:", last)

df = pd.read_csv(find(CSV), low_memory=False)
df = df[df[LAT].between(41, 84) & df[LON].between(-142, -52)].dropna(subset=[TARGET]).copy()
df["_d"] = pd.to_datetime(df["REP_DATE"].astype(str).str[:10], errors="coerce")
df = df[df["_d"].between("2004-01-01", pd.Timestamp(last) - pd.Timedelta(days=3))].copy()
df["_id"] = df.index.astype(str)
train, test = df[df["year"].isin(TRAIN_YEARS)], df[df["year"].isin(TEST_YEARS)]
use = pd.concat([train, test]).drop_duplicates("_id")
print(f"train years {TRAIN_YEARS}: {len(train):,} fires | test years {TEST_YEARS}: {len(test):,} fires")

done = set(pd.read_csv(OUT)["_id"].astype(str)) if os.path.exists(OUT) else set()
todo = use[~use["_id"].isin(done)]; print(f"already fetched {len(done):,}; remaining {len(todo):,}")

def per_fire(f):
    d0 = ee.Date(f.get("ds")); col = ERA.filterDate(d0, d0.advance(3, "day"))
    wind = col.map(lambda im: im.expression("sqrt(u*u+v*v)", {"u": im.select("u_component_of_wind_10m"),
                                                              "v": im.select("v_component_of_wind_10m")}).rename("w"))
    rh = col.map(lambda im: im.expression(
        "min(100, 100*exp(17.625*Td/(243.04+Td) - 17.625*T/(243.04+T)))",
        {"T": im.select("temperature_2m").subtract(273.15), "Td": im.select("dewpoint_temperature_2m").subtract(273.15)}).rename("rh"))
    img = ee.Image.cat([
        col.select("temperature_2m_max").max().subtract(273.15).rename("t"),
        wind.max().rename("wmax"), wind.mean().rename("wmean"),
        col.select("total_precipitation_sum").sum().multiply(1000).rename("p"),
        rh.min().rename("rh")])
    return f.set(img.reduceRegion(ee.Reducer.first(), f.geometry(), 11132))

t0 = time.time(); n_done = 0; n_todo = len(todo)
for i in range(0, n_todo, BATCH):
    b = todo.iloc[i:i + BATCH]
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(lo), float(la)]),
                                          {"_id": str(k), "ds": d.strftime("%Y-%m-%d")})
                               for k, lo, la, d in zip(b["_id"], b[LON], b[LAT], b["_d"])])
    for attempt in range(4):
        try:
            res = fc.map(per_fire).getInfo()["features"]; break
        except Exception as e:
            print("  retry after error:", str(e)[:120]); time.sleep(10 * (attempt + 1)); res = None
    if res is None: print("giving up on this batch; rerun later"); continue
    rows = [[r["properties"]["_id"], r["properties"].get("t"), r["properties"].get("wmax"), r["properties"].get("wmean"),
             r["properties"].get("p"), r["properties"].get("rh")] for r in res]
    pd.DataFrame(rows, columns=["_id"] + NEW).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
    n_done += len(rows); rate = n_done / max(time.time() - t0, 1)
    print(f"progress: {n_done:,}/{n_todo:,} fires ({n_done / max(n_todo, 1):.0%}) | {rate:.1f} fires/s | about {(n_todo - n_done) / max(rate, 1e-9) / 60:.0f} min left", flush=True)

got = pd.read_csv(OUT).drop_duplicates("_id"); got["_id"] = got["_id"].astype(str)
m = use.merge(got, on="_id", how="inner").dropna(subset=NEW)
if len(m) < 0.97 * len(use): raise SystemExit(f"Only {len(m):,} of {len(use):,} fires have weather. Run again to continue.")

feats = json.load(open(find(INFO)))["features"]
tr, te = m[m["_id"].isin(set(train["_id"]))], m[m["_id"].isin(set(test["_id"]))]
PARAMS = dict(n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=31, is_unbalance=True, random_state=42, verbose=-1)
ytr, yte = tr[TARGET].astype(int).values, te[TARGET].astype(int).values
print(f"\ntrain {len(tr):,} | test {len(te):,} ({yte.mean():.3f} big)")
sA = LGBMClassifier(**PARAMS).fit(tr[feats], ytr).predict_proba(te[feats])[:, 1]
sB = LGBMClassifier(**PARAMS).fit(tr[feats + NEW], ytr).predict_proba(te[feats + NEW])[:, 1]
r = np.random.RandomState(1); dA, dP = [], []
for _ in range(500):
    i = r.randint(0, len(yte), len(yte))
    if yte[i].sum() == 0: continue
    dA.append(roc_auc_score(yte[i], sB[i]) - roc_auc_score(yte[i], sA[i]))
    dP.append(average_precision_score(yte[i], sB[i]) - average_precision_score(yte[i], sA[i]))
for lab, s in (("Without next-3-day weather (20 features)", sA), ("With next-3-day ACTUAL weather, Earth Engine (25 features)", sB)):
    order = np.argsort(-s); top = yte[order[:int(.15 * len(s))]].sum() / yte.sum()
    print(f"{lab:<58} ROC-AUC {roc_auc_score(yte, s):.3f} | PR-AUC {average_precision_score(yte, s):.3f} | top 15% catches {top:.0%}")
print(f"Gain in ROC-AUC {np.mean(dA):+.3f} (95% {np.percentile(dA,2.5):+.3f} to {np.percentile(dA,97.5):+.3f})")
print(f"Gain in PR-AUC  {np.mean(dP):+.3f} (95% {np.percentile(dP,2.5):+.3f} to {np.percentile(dP,97.5):+.3f})")
print("\nBest case (perfect forecast). A real forecast gains less. Done. Paste all of this output back.")
