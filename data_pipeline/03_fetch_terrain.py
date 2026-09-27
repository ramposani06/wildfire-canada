"""
Fetch terrain (elevation, slope) and vegetation (NDVI) for a list of
fires using Google Earth Engine.

Important: SRTM (USGS/SRTMGL1_003) has no coverage above ~60°N. Any
fire north of that latitude needs a fallback source for BOTH elevation
and slope. A past version of this pipeline implemented the fallback for
elevation but not slope — every high-latitude fire silently got a
near-zero slope value instead of a real one. That bug lived in the
production model's training data for a long time before being caught by
manually re-querying a handful of known fires and comparing. Keep both
fallbacks wired up together; never add one without the other.
"""
import ee
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

ee.Initialize()

SRTM = ee.Image("USGS/SRTMGL1_003")
SLOPE_SRTM = ee.Terrain.slope(SRTM)

# Fallback for above ~60°N, where SRTM has no data.
DEM_FALLBACK = ee.ImageCollection("COPERNICUS/DEM/GLO30").mosaic().select("DEM")
SLOPE_FALLBACK = ee.Terrain.slope(DEM_FALLBACK)

NDVI_COLLECTION = "MODIS/061/MOD13Q1"


def fetch_terrain_one(lat: float, lon: float) -> dict:
    pt = ee.Geometry.Point([lon, lat])

    elev = SRTM.reduceRegion(ee.Reducer.mean(), pt, 30).get("elevation").getInfo()
    if elev is None:
        elev = DEM_FALLBACK.reduceRegion(ee.Reducer.mean(), pt, 30).get("DEM").getInfo()
        slope_val = SLOPE_FALLBACK.reduceRegion(ee.Reducer.mean(), pt, 30).get("slope").getInfo()
    else:
        slope_val = SLOPE_SRTM.reduceRegion(ee.Reducer.mean(), pt, 30).get("slope").getInfo()

    return {"elevation": elev, "slope": slope_val}


def fetch_ndvi_one(lat: float, lon: float, date: pd.Timestamp) -> float:
    pt = ee.Geometry.Point([lon, lat])
    window_start = (date - pd.Timedelta(days=16)).strftime("%Y-%m-%d")
    window_end = date.strftime("%Y-%m-%d")
    img = (ee.ImageCollection(NDVI_COLLECTION)
           .filterDate(window_start, window_end)
           .select("NDVI")
           .mosaic())
    val = img.reduceRegion(ee.Reducer.mean(), pt, 250).get("NDVI").getInfo()
    return val / 10000.0 if val is not None else None  # MOD13Q1 NDVI is scaled by 10000


def fetch_terrain(fires: pd.DataFrame, max_workers: int = 8) -> pd.DataFrame:
    """
    fires must have columns: UNIQUE_ID, LATITUDE, LONGITUDE, REP_DATE.
    Returns a DataFrame with UNIQUE_ID, elevation, slope, NDVI.
    """
    rows = []

    def worker(row):
        try:
            terrain = fetch_terrain_one(row.LATITUDE, row.LONGITUDE)
            ndvi = fetch_ndvi_one(row.LATITUDE, row.LONGITUDE, row.REP_DATE)
            return {"UNIQUE_ID": row.UNIQUE_ID, **terrain, "NDVI": ndvi}
        except Exception:
            return {"UNIQUE_ID": row.UNIQUE_ID, "elevation": None, "slope": None, "NDVI": None}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(worker, row) for row in fires.itertuples()]
        for f in as_completed(futures):
            rows.append(f.result())

    return pd.DataFrame(rows)


if __name__ == "__main__":
    fires = pd.read_csv("fires_to_fetch.csv", parse_dates=["REP_DATE"])
    result = fetch_terrain(fires)
    result.to_csv("terrain_output.csv", index=False)
    print(f"Fetched terrain for {len(result):,} fires")
