# Test: LONG-MEMORY dryness and snow. The model only sees weather for the 7 days before the report. Big fires also depend on how dry the
# ground and fuel have been for weeks (soil moisture, rain over 30-90 days, days since last rain, recent snow melt). Spring misses (May, cool, dry grass) may fit this.
# Data: ERA5-Land daily (Google Earth Engine), only days BEFORE the report day (nothing from the report day). Resumable: results append to dryness_raw.csv on Drive.
# Test protocol: 22 inputs vs 22 + new columns on two forward splits (train<=2021 -> 2022-24, train<=2024 -> 2025+), bootstrap ranges.
import os, json, time, warnings
import numpy as np, pandas as pd, ee
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive"); CSV = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
PROJECT = os.environ.get("EE_PROJECT", "project-8d6e8459-b76f-4a38-a85"); BATCH = 500
OUT = os.path.join(FOLDER, "dryness_raw.csv"); LAT, LON, TARGET = "LATITUDE", "LONGITUDE", "is_big_fire"
def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files: return os.path.join(root, name)
    raise FileNotFoundError(name)
ee.Initialize(project=PROJECT); ERA = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
last = ee.Date(ERA.aggregate_max("system:time_start")).format("YYYY-MM-dd").getInfo(); print("ERA5-Land daily data runs to:", last)
df = pd.read_csv(find(CSV), low_memory=False)
df = df[df[LAT].between(41, 84) & df[LON].between(-142, -52)].dropna(subset=[TARGET]).copy()
df["_d"] = pd.to_datetime(df["REP_DATE"].astype(str).str[:10], errors="coerce"); df = df[df["_d"].between("2004-01-01", pd.Timestamp(last))].copy(); df["_id"] = df.index.astype(str)
done = set(pd.read_csv(OUT)["_id"].astype(str)) if os.path.exists(OUT) else set()
todo = df[~df["_id"].isin(done)].sort_values("_d", ascending=False); print(f"fires {len(df):,}; already fetched {len(done):,}; remaining {len(todo):,}")
NAMES = ["soil1", "soil2", "soil3", "p14", "p30", "p60", "p90", "pet30", "tmean30", "dsr", "snow30", "snow90"]
def per_fire(f):
    d0 = ee.Date(f.get("ds")); w = lambda n: ERA.filterDate(d0.advance(-n, "day"), d0)
    c3, c7, c14, c30, c60, c90 = w(3), w(7), w(14), w(30), w(60), w(90)
    def rain_k(im):
        k = d0.difference(ee.Date(im.get("system:time_start")), "day")
        return ee.Image.constant(k).toFloat().updateMask(im.select("total_precipitation_sum").gt(0.002)).rename("dsr")
    img = ee.Image.cat([
        c3.select("volumetric_soil_water_layer_1").mean().rename("soil1"), c7.select("volumetric_soil_water_layer_2").mean().rename("soil2"), c14.select("volumetric_soil_water_layer_3").mean().rename("soil3"),
        c14.select("total_precipitation_sum").sum().multiply(1000).rename("p14"), c30.select("total_precipitation_sum").sum().multiply(1000).rename("p30"),
        c60.select("total_precipitation_sum").sum().multiply(1000).rename("p60"), c90.select("total_precipitation_sum").sum().multiply(1000).rename("p90"),
        c30.select("potential_evaporation_sum").sum().multiply(1000).rename("pet30"), c30.select("temperature_2m").mean().subtract(273.15).rename("tmean30"),
        c60.map(rain_k).min().unmask(61).rename("dsr"),
        c30.select("snow_cover").map(lambda im: im.gt(50)).sum().rename("snow30"), c90.select("snow_cover").map(lambda im: im.gt(50)).sum().rename("snow90")])
    return f.set(img.reduceRegion(ee.Reducer.first(), f.geometry(), 11132))
t0 = time.time(); n_done = 0; n_todo = len(todo)
for i in range(0, n_todo, BATCH):
    b = todo.iloc[i:i + BATCH]
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(lo), float(la)]), {"_id": str(k), "ds": d.strftime("%Y-%m-%d")}) for k, lo, la, d in zip(b["_id"], b[LON], b[LAT], b["_d"])])
    res = None
    for attempt in range(4):
        try: res = fc.map(per_fire).getInfo()["features"]; break
        except Exception as e: print("  retry after error:", str(e)[:120]); time.sleep(10 * (attempt + 1))
    if res is None: print("giving up on this batch; rerun later"); continue
    pd.DataFrame([[r["properties"]["_id"]] + [r["properties"].get(k) for k in NAMES] for r in res], columns=["_id"] + NAMES).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
    n_done += len(res); rate = n_done / max(time.time() - t0, 1)
    print(f"progress: {n_done:,}/{n_todo:,} ({n_done / max(n_todo, 1):.0%}) | {rate:.1f} fires/s | about {(n_todo - n_done) / max(rate, 1e-9) / 60:.0f} min left", flush=True)
raw = pd.read_csv(OUT).drop_duplicates("_id"); raw["_id"] = raw["_id"].astype(str)
m = df.merge(raw, on="_id", how="left")
if m["p30"].notna().mean() < 0.97: raise SystemExit(f"Only {m['p30'].notna().mean():.0%} of fires have values. Run again to continue.")
m["dry_ratio30"] = m["p30"] / (m["pet30"].abs() + 1.0); m["dry_ratio90"] = m["p90"] / (m["pet30"].abs() * 3 + 1.0)
GROUPS = {"soil moisture (3 layers)": ["soil1", "soil2", "soil3"], "rain memory (14-90 days, days since rain)": ["p14", "p30", "p60", "p90", "dsr", "dry_ratio30", "dry_ratio90"],
          "snow (days snow-covered, 30 and 90)": ["snow30", "snow90"], "30-day heat and evaporation demand": ["tmean30", "pet30"]}
ALL = [c for g in GROUPS.values() for c in g]
base = json.load(open(find("final_model_v14.6_nosat_info.json")))["features"] + [LAT, LON]
P = dict(n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=31, is_unbalance=True, random_state=42, verbose=-1)
y = m[TARGET].astype(int).values; yr = m["year"].values; mon = m["_d"].dt.month.values; rng = np.random.default_rng(0)
def ci(v): return f"{np.mean(v):+.3f} ({np.percentile(v,2.5):+.3f} to {np.percentile(v,97.5):+.3f})"
def top15(s, yy): o = np.argsort(-s)[:int(.15 * len(s))]; return yy[o].sum() / yy.sum()
def run(name, trm, tem):
    tri, tei = np.where(trm)[0], np.where(tem)[0]; yt = y[tei]; print(f"\n######## {name}: train {len(tri):,} | test {len(tei):,} fires, {yt.sum():,} big ########")
    s0 = LGBMClassifier(**P).fit(m.iloc[tri][base], y[tri]).predict_proba(m.iloc[tei][base])[:, 1]; print(f"baseline (22 inputs)  ROC {roc_auc_score(yt,s0):.3f} PR {average_precision_score(yt,s0):.3f} top15% {top15(s0,yt):.0%}")
    sp = (mon[tei] >= 4) & (mon[tei] <= 5)
    for gname, cols in list(GROUPS.items()) + [("ALL new columns", ALL)]:
        s1 = LGBMClassifier(**P).fit(m.iloc[tri][base + cols], y[tri]).predict_proba(m.iloc[tei][base + cols])[:, 1]; d, dr, ds = [], [], []
        for _ in range(500):
            b = rng.integers(0, len(tei), len(tei)); yb = yt[b]
            if yb.sum() == 0: continue
            d.append(average_precision_score(yb, s1[b]) - average_precision_score(yb, s0[b])); dr.append(roc_auc_score(yb, s1[b]) - roc_auc_score(yb, s0[b]))
        print(f"  + {gname:42s} ROC {roc_auc_score(yt,s1):.3f} PR {average_precision_score(yt,s1):.3f} | PR gain {ci(d)} | ROC gain {ci(dr)} | top15% {top15(s1,yt):.0%} | Apr-May PR {average_precision_score(yt[sp],s1[sp]):.3f} vs {average_precision_score(yt[sp],s0[sp]):.3f}")
run("train<=2021 -> 2022-24", yr <= 2021, (yr >= 2022) & (yr <= 2024)); run("train<=2024 -> 2025+", yr <= 2024, yr >= 2025)
print("\nDone. Paste all of this output back.")
