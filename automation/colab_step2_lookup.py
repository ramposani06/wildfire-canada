# STEP 2 for automation (run in Colab with %run). Builds a small lookup grid for
#   - distance to nearest road (from nrn_roads_merged_fixed.gpkg)
#   - people within 10 km and 25 km (from kontur_pop_ca.gpkg)
# then CHECKS it against the real values stored for past fires.
# Output: MyDrive/wildfire_project/automation/road_pop_grid.npz  (a few MB)
# Resumable: road distance is saved after every latitude band; run again to continue.
# Takes roughly 20-40 minutes. Keep the tab open.
import os, glob, time, math
import numpy as np, pandas as pd
import geopandas as gpd, shapely, pyogrio
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
PART = f"{OUT_DIR}/dist_partial.npz"

LAT0, LAT1, LON0, LON1 = 41.0, 70.0, -142.0, -52.0
RES_D, RES_P = 0.02, 0.05
TILE_LAT, TILE_LON = 100, 150            # grid cells per tile (2.0 x 3.0 degrees)
MARGINS = [0.5, 2.0, 6.0, 15.0]          # degrees of extra roads to read, tried in order
CAP_M = 600_000                          # unresolved points get this (lower bound), fits in uint16 x 10 m
t0 = time.time()
def log(*a): print(f"[{(time.time()-t0)/60:5.1f} min]", *a, flush=True)

to3347 = Transformer.from_crs(4326, 3347, always_xy=True)
layer = pyogrio.list_layers(ROADS)[0][0]
roads_crs = CRS.from_user_input(pyogrio.read_info(ROADS, layer=layer)["crs"])
to_roads = Transformer.from_crs(4326, roads_crs, always_xy=True)
log("roads layer:", layer, "| crs:", roads_crs.to_string())

def read_roads(lon_lo, lat_lo, lon_hi, lat_hi):
    bb = to_roads.transform_bounds(lon_lo, lat_lo, lon_hi, lat_hi, densify_pts=21)
    try:
        g = gpd.read_file(ROADS, layer=layer, bbox=tuple(bb), engine="pyogrio", columns=[])
    except Exception:
        g = gpd.read_file(ROADS, layer=layer, bbox=tuple(bb), engine="pyogrio")
    if len(g) == 0: return None
    g = g[g.geometry.notna()]
    if roads_crs.to_epsg() != 3347: g = g.to_crs(3347)
    return g.geometry.values

# ------------------------------------------------------------ road distance
n_lat = int(round((LAT1 - LAT0) / RES_D)); n_lon = int(round((LON1 - LON0) / RES_D))
dist = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
done_bands = 0
if os.path.exists(PART):
    z = np.load(PART); dist = z["dist"]; done_bands = int(z["done"]); log("resuming at band", done_bands)

def tile_dist(lat_c, lon_c):
    x, y = to3347.transform(lon_c, lat_c)
    pts = shapely.points(x, y)
    out = np.full(len(x), np.nan)
    pending = np.arange(len(x))
    for margin in MARGINS:
        if len(pending) == 0: break
        lat_lo, lat_hi = lat_c.min() - margin, lat_c.max() + margin
        pad = margin / max(math.cos(math.radians(min(80.0, abs(lat_hi)))), 0.2)
        geoms = read_roads(lon_c.min() - pad, lat_lo, lon_c.max() + pad, lat_hi)
        if geoms is None: continue
        tree = shapely.STRtree(geoms)
        idx = tree.nearest(pts[pending])
        d = shapely.distance(pts[pending], geoms[idx])
        safe = margin * 111_000 * 0.8                 # nearest road is surely inside the window
        ok = d <= safe
        out[pending[ok]] = d[ok]
        pending = pending[~ok]
    out[np.isnan(out)] = CAP_M
    return out

n_bands = n_lat // TILE_LAT + (n_lat % TILE_LAT > 0)
for b in range(done_bands, n_bands):
    i0, i1 = b * TILE_LAT, min((b + 1) * TILE_LAT, n_lat)
    for j0 in range(0, n_lon, TILE_LON):
        j1 = min(j0 + TILE_LON, n_lon)
        ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
        lat_c = LAT0 + (ii.ravel() + 0.5) * RES_D
        lon_c = LON0 + (jj.ravel() + 0.5) * RES_D
        dist[i0:i1, j0:j1] = tile_dist(lat_c, lon_c).reshape(ii.shape)
    np.savez(PART, dist=dist, done=b + 1)
    log(f"road band {b+1}/{n_bands} done (lat {LAT0 + i0*RES_D:.1f} to {LAT0 + i1*RES_D:.1f})")
dist_10m = np.clip(np.round(dist / 10.0), 0, 65535).astype(np.uint16)

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
