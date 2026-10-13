# STEP 2 for automation (run in Colab with %run). Builds a small lookup grid for
#   - distance to nearest road (from nrn_roads_merged_fixed.gpkg)
#   - people within 10 km and 25 km (from kontur_pop_ca.gpkg)
# then CHECKS it against the real values stored for past fires.
# Output: MyDrive/wildfire_project/automation/road_pop_grid.npz  (a few MB)
# Resumable: the road points are saved to Drive after the first (slow) step; run again to skip it.
# Takes roughly 10-25 minutes. Keep the tab open.
import os, glob, time, math
import numpy as np, pandas as pd
import geopandas as gpd, shapely, pyogrio
from pyogrio.raw import read as raw_read
from pyproj import CRS, Transformer
from scipy.signal import fftconvolve
from scipy.stats import spearmanr
try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception:
    pass

DRIVE = "/content/drive/MyDrive"
def find(name):
    hits = glob.glob(f"{DRIVE}/**/{name}", recursive=True)
    if not hits: raise FileNotFoundError(name)
    return hits[0]

ROADS = find("nrn_roads_merged_fixed.gpkg")
POP = find("kontur_pop_ca.gpkg")
CSV = find("unified_dataset_2004_2026_FINAL_v4.csv")
OUT_DIR = f"{DRIVE}/wildfire_project/automation"; os.makedirs(OUT_DIR, exist_ok=True)
OUT = f"{OUT_DIR}/road_pop_grid.npz"

LAT0, LAT1, LON0, LON1 = 41.0, 70.0, -142.0, -52.0
RES_D, RES_P = 0.02, 0.05
t0 = time.time()
def log(*a): print(f"[{(time.time()-t0)/60:5.1f} min]", *a, flush=True)

to3347 = Transformer.from_crs(4326, 3347, always_xy=True)
layer = pyogrio.list_layers(ROADS)[0][0]
roads_crs = CRS.from_user_input(pyogrio.read_info(ROADS, layer=layer)["crs"])
n_feat = int(pyogrio.read_info(ROADS, layer=layer)["features"])
log("roads layer:", layer, "| crs:", roads_crs.to_string(), "| road pieces:", f"{n_feat:,}")

# ------------------------------------------------------------ road distance
# Fast method: turn every road into points about 500 m apart (read in chunks so memory stays small),
# then ask a KD-tree for the nearest road point of every grid cell. Error is a few hundred metres at most.
n_lat = int(round((LAT1 - LAT0) / RES_D)); n_lon = int(round((LON1 - LON0) / RES_D))
PTS = f"{OUT_DIR}/road_points_3347.npy"
if os.path.exists(PTS):
    road_xy = np.load(PTS); log(f"loaded saved road points: {len(road_xy):,}")
else:
    CHUNK = 150_000
    to_3347_from_roads = None if roads_crs.to_epsg() == 3347 else Transformer.from_crs(roads_crs, 3347, always_xy=True)
    parts = []
    for start in range(0, n_feat, CHUNK):
        try:
            _, _, geom, _ = raw_read(ROADS, layer=layer, columns=[], skip_features=start, max_features=CHUNK)
        except Exception:
            _, _, geom, _ = raw_read(ROADS, layer=layer, skip_features=start, max_features=CHUNK)
        g = shapely.from_wkb(geom)
        g = g[~shapely.is_missing(g)]
        g = g[~shapely.is_empty(g)]
        xy = shapely.get_coordinates(shapely.segmentize(g, 500.0))
        if to_3347_from_roads is not None:
            x, y = to_3347_from_roads.transform(xy[:, 0], xy[:, 1]); xy = np.column_stack([x, y])
        key = pd.DataFrame({"a": (xy[:, 0] // 250).astype(np.int64), "b": (xy[:, 1] // 250).astype(np.int64)})
        parts.append(xy[~key.duplicated().values].astype(np.float32))
        log(f"roads read {min(start + CHUNK, n_feat):,}/{n_feat:,}")
        del g, xy, key
    road_xy = np.vstack(parts); del parts
    key = pd.DataFrame({"a": (road_xy[:, 0] // 250).astype(np.int64), "b": (road_xy[:, 1] // 250).astype(np.int64)})
    road_xy = road_xy[~key.duplicated().values]
    np.save(PTS, road_xy); log(f"road points kept: {len(road_xy):,} (saved)")

from scipy.spatial import cKDTree
tree = cKDTree(road_xy)
lat_c = LAT0 + (np.arange(n_lat) + 0.5) * RES_D
lon_c = LON0 + (np.arange(n_lon) + 0.5) * RES_D
dist = np.empty((n_lat, n_lon), dtype=np.float32)
LOg = np.broadcast_to(lon_c, (n_lat, n_lon))
for i in range(0, n_lat, 50):
    la_blk = np.broadcast_to(lat_c[i:i + 50, None], (min(50, n_lat - i), n_lon))
    gx, gy = to3347.transform(LOg[i:i + 50].ravel(), la_blk.ravel())
    d, _ = tree.query(np.column_stack([gx, gy]), k=1, workers=-1)
    dist[i:i + 50] = d.reshape(-1, n_lon)
log("road distance grid done")
dist_10m = np.clip(np.round(dist / 10.0), 0, 65535).astype(np.uint16)
del tree

# --------------------------------------------------------------- population
log("reading population hexagons ...")
pg = gpd.read_file(POP)
if "population" not in pg.columns:
    raise KeyError(f"no 'population' column; columns are {list(pg.columns)}")
pg = pg.to_crs(3347)
c = pg.geometry.centroid
px, py, pv = c.x.values, c.y.values, pg["population"].values.astype("float32")
xmin, ymin = math.floor(px.min() / 1000) - 30, math.floor(py.min() / 1000) - 30
W = int(math.ceil(px.max() / 1000)) + 30 - xmin; H = int(math.ceil(py.max() / 1000)) + 30 - ymin
ras = np.zeros((H, W), dtype=np.float32)
np.add.at(ras, ((py / 1000).astype(int) - ymin, (px / 1000).astype(int) - xmin), pv)
log(f"population raster {W}x{H} km, total people {ras.sum():,.0f}")
def disk(r):
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy <= r * r).astype(np.float32)
s10 = fftconvolve(ras, disk(10), mode="same"); log("10 km sums done")
s25 = fftconvolve(ras, disk(25), mode="same"); log("25 km sums done")
del ras

n_latp = int(round((LAT1 - LAT0) / RES_P)); n_lonp = int(round((LON1 - LON0) / RES_P))
la = LAT0 + (np.arange(n_latp) + 0.5) * RES_P; lo = LON0 + (np.arange(n_lonp) + 0.5) * RES_P
LO, LA = np.meshgrid(lo, la)
gx, gy = to3347.transform(LO.ravel(), LA.ravel())
gi = np.clip((gy / 1000).astype(int) - ymin, 0, H - 1); gj = np.clip((gx / 1000).astype(int) - xmin, 0, W - 1)
pop10 = np.maximum(s10[gi, gj], 0).reshape(n_latp, n_lonp).astype(np.float32)
pop25 = np.maximum(s25[gi, gj], 0).reshape(n_latp, n_lonp).astype(np.float32)
del s10, s25

np.savez_compressed(OUT, dist_10m=dist_10m, pop10=pop10, pop25=pop25,
                    meta=np.array([LAT0, LAT1, LON0, LON1, RES_D, RES_P]))
log(f"saved {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")

# ------------------------------------------------- check against past fires
log("checking against real values stored for past fires ...")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "/content/wildfire-canada/automation")
from road_pop_lookup import RoadPopLookup
L = RoadPopLookup(OUT)
df = pd.read_csv(CSV, low_memory=False, usecols=["LATITUDE", "LONGITUDE", "dist_to_road_m", "pop_within_10km", "pop_within_25km"]).dropna()
df = df[df.LATITUDE.between(41, 69.9) & df.LONGITUDE.between(-141.9, -52.1)].sample(6000, random_state=1)
got = L.get_many(df.LATITUDE.values, df.LONGITUDE.values)
d_true, d_got = df.dist_to_road_m.values, got[:, 0]
print("\n=== ROAD DISTANCE: lookup vs real value (6,000 past fires) ===")
print(f"rank match (Spearman): {spearmanr(d_true, d_got)[0]:.3f}")
print(f"median error: {np.median(np.abs(d_true - d_got)):.0f} m | within 1 km: {np.mean(np.abs(d_true - d_got) <= 1000)*100:.0f}% | within 3 km: {np.mean(np.abs(d_true - d_got) <= 3000)*100:.0f}%")
for k, col in [(1, "pop_within_10km"), (2, "pop_within_25km")]:
    t, g = df[col].values, got[:, k]
    big = t > 100
    print(f"\n=== {col.upper()} ===")
    print(f"rank match (Spearman): {spearmanr(t, g)[0]:.3f} | median relative error where people > 100: {np.median(np.abs(g[big]-t[big])/t[big])*100:.0f}%")
log("done. Paste all of this output back.")
