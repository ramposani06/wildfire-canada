# Test: aspect (which way the slope faces) + wind direction + "wind blowing uphill" features.
# Wind: ERA5-Land daily mean wind (u, v) for the 3 whole days BEFORE the report day (nothing from the report day).
# Aspect: Copernicus 30 m DEM. Both come from Google Earth Engine. Run in Colab after ee.Authenticate().
# Train on TRAIN_YEARS (default 2022-24), test on TEST_YEARS (default 2025+). Resumable (results append to a CSV on Drive).
import os, json, time, warnings
import numpy as np, pandas as pd, ee
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive")
CSV    = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
INFO   = "final_model_v14.6_nosat_info.json"
PROJECT = os.environ.get("EE_PROJECT", "project-8d6e8459-b76f-4a38-a85")
TRAIN_YEARS = [int(x) for x in os.environ.get("WF_TRAIN_YEARS", "2022,2023,2024").split(",")]
TEST_YEARS  = [int(x) for x in os.environ.get("WF_TEST_YEARS", "2025,2026").split(",")]
BATCH = 1000
OUT = os.path.join(FOLDER, "aspect_wind_raw.csv")
LAT, LON, TARGET = "LATITUDE", "LONGITUDE", "is_big_fire"
def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files: return os.path.join(root, name)
    raise FileNotFoundError(name)

ee.Initialize(project=PROJECT)
ERA = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
last = ee.Date(ERA.aggregate_max("system:time_start")).format("YYYY-MM-dd").getInfo()
print("ERA5-Land daily data runs to:", last)
c = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM")
dem = c.mosaic().setDefaultProjection(c.first().projection())
asp = ee.Terrain.aspect(dem).multiply(np.pi / 180)
ASPECT = ee.Image.cat([asp.sin().rename("asp_sin"), asp.cos().rename("asp_cos")])

df = pd.read_csv(find(CSV), low_memory=False)
df = df[df[LAT].between(41, 84) & df[LON].between(-142, -52)].dropna(subset=[TARGET]).copy()
df["_d"] = pd.to_datetime(df["REP_DATE"].astype(str).str[:10], errors="coerce")
df = df[df["_d"].between("2004-01-01", pd.Timestamp(last))].copy(); df["_id"] = df.index.astype(str)
train, test = df[df["year"].isin(TRAIN_YEARS)], df[df["year"].isin(TEST_YEARS)]
use = pd.concat([train, test]).drop_duplicates("_id")
print(f"train {TRAIN_YEARS}: {len(train):,} fires | test {TEST_YEARS}: {len(test):,} fires")
done = set(pd.read_csv(OUT)["_id"].astype(str)) if os.path.exists(OUT) else set()
todo = use[~use["_id"].isin(done)]; print(f"already fetched {len(done):,}; remaining {len(todo):,}")

def per_fire(f):
    d0 = ee.Date(f.get("ds")); col = ERA.filterDate(d0.advance(-3, "day"), d0)
    spd = col.map(lambda im: im.expression("sqrt(u*u+v*v)", {"u": im.select("u_component_of_wind_10m"),
                                                             "v": im.select("v_component_of_wind_10m")}).rename("s"))
    w = ee.Image.cat([col.select("u_component_of_wind_10m").mean().rename("u"),
                      col.select("v_component_of_wind_10m").mean().rename("v"), spd.mean().rename("spd")])
    f = f.set(w.reduceRegion(ee.Reducer.first(), f.geometry(), 11132))
    return f.set(ASPECT.reduceRegion(ee.Reducer.first(), f.geometry(), 30))

t0 = time.time(); n_done = 0; n_todo = len(todo)
for i in range(0, n_todo, BATCH):
    b = todo.iloc[i:i + BATCH]
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(lo), float(la)]), {"_id": str(k), "ds": d.strftime("%Y-%m-%d")})
                               for k, lo, la, d in zip(b["_id"], b[LON], b[LAT], b["_d"])])
    res = None
    for attempt in range(4):
        try: res = fc.map(per_fire).getInfo()["features"]; break
        except Exception as e: print("  retry after error:", str(e)[:120]); time.sleep(10 * (attempt + 1))
    if res is None: print("giving up on this batch; rerun later"); continue
    rows = [[r["properties"]["_id"]] + [r["properties"].get(k) for k in ("u", "v", "spd", "asp_sin", "asp_cos")] for r in res]
    pd.DataFrame(rows, columns=["_id", "u", "v", "spd", "asp_sin", "asp_cos"]).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
    n_done += len(rows); rate = n_done / max(time.time() - t0, 1)
    print(f"progress: {n_done:,}/{n_todo:,} ({n_done / max(n_todo, 1):.0%}) | {rate:.1f} fires/s | about {(n_todo - n_done) / max(rate, 1e-9) / 60:.0f} min left", flush=True)

raw = pd.read_csv(OUT).drop_duplicates("_id"); raw["_id"] = raw["_id"].astype(str)
m = use.merge(raw, on="_id", how="inner").dropna(subset=["u", "v", "spd", "asp_sin", "asp_cos"])
if len(m) < 0.97 * len(use): raise SystemExit(f"Only {len(m):,} of {len(use):,} fires have values. Run again to continue.")
# derived columns. wind_to = direction the wind blows TOWARD (clockwise from north). aspect = direction the slope faces (downhill).
m["wind_to"] = np.arctan2(m["u"], m["v"]); m["asp"] = np.arctan2(m["asp_sin"], m["asp_cos"])
m["wind_dir_sin"], m["wind_dir_cos"] = np.sin(m["wind_to"]), np.cos(m["wind_to"])
m["wind_vec_speed"] = np.sqrt(m["u"] ** 2 + m["v"] ** 2); m["wind_steadiness"] = m["wind_vec_speed"] / m["spd"].clip(lower=0.1)
m["upslope_wind"] = -np.cos(m["wind_to"] - m["asp"]) * m["wind_vec_speed"]       # positive = blowing uphill
m["upslope_x_slope"] = m["upslope_wind"] * m["slope"]
NEW = ["asp_sin", "asp_cos", "wind_dir_sin", "wind_dir_cos", "wind_vec_speed", "wind_steadiness", "upslope_wind", "upslope_x_slope"]
print("\nBig-fire rate by wind relative to slope (test years, slope > 5 degrees):")
t = m[m["_id"].isin(set(test["_id"])) & (m["slope"] > 5)]
print(t.groupby(pd.cut(t["upslope_wind"], [-99, -1, 0, 1, 99], labels=["downhill >1", "downhill", "uphill", "uphill >1"]))[TARGET].agg(["size", "mean"]).round(3).to_string())

base = json.load(open(find(INFO)))["features"]
P = dict(n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=31, is_unbalance=True, random_state=42, verbose=-1)
tr, te = m[m["_id"].isin(set(train["_id"]))], m[m["_id"].isin(set(test["_id"]))]
ytr, yte = tr[TARGET].astype(int).values, te[TARGET].astype(int).values
print(f"\ntrain {len(tr):,} | test {len(te):,} ({yte.mean():.3f} big)")
def fit(fs): return LGBMClassifier(**P).fit(tr[fs], ytr).predict_proba(te[fs])[:, 1]
def compare(name, A, B):
    r = np.random.RandomState(1); dA, dP = [], []
    for _ in range(500):
        i = r.randint(0, len(yte), len(yte))
        if yte[i].sum() == 0: continue
        dA.append(roc_auc_score(yte[i], B[i]) - roc_auc_score(yte[i], A[i])); dP.append(average_precision_score(yte[i], B[i]) - average_precision_score(yte[i], A[i]))
    print(f"{name}: without ROC {roc_auc_score(yte, A):.3f} PR {average_precision_score(yte, A):.3f} | with ROC {roc_auc_score(yte, B):.3f} PR {average_precision_score(yte, B):.3f}")
    print(f"   gain ROC {np.mean(dA):+.3f} ({np.percentile(dA,2.5):+.3f} to {np.percentile(dA,97.5):+.3f}) | PR {np.mean(dP):+.3f} ({np.percentile(dP,2.5):+.3f} to {np.percentile(dP,97.5):+.3f})")
sA, sB = fit(base), fit(base + NEW); compare("20 features + 8 aspect/wind columns", sA, sB)
loc = base + [LAT, LON]; sC, sD = fit(loc), fit(loc + NEW); compare("22 features (v14.7) + 8 aspect/wind columns", sC, sD)
sE = fit(base + ["asp_sin", "asp_cos"]); compare("20 features + aspect only", sA, sE)
print("\nDone. Paste all of this output back.")
