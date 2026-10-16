"""Daily job: score newly reported fires with the frozen v14.7 model and keep a log.

Each run:
  1. downloads the current active-fire list (Canadian Wildland Fire Information System)
  2. scores every fire that is new on the list (not in the log yet); the date it first appears is its report date.
     The very first run only records the fires already burning (no scoring), because their real report dates are unknown
  3. updates size and status of fires already in the log, so we learn which became big
  4. writes automation/data/scored_log.csv and automation/data/latest_run.json

The model is never changed here. The log is the clean forward test for 2027.
Needs (GitHub secret): EE_SERVICE_ACCOUNT_JSON  (Earth Engine key for NDVI, elevation, slope)
"""
import json
import os
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "modeling"))
from feature_config import PROVINCE_MAP  # noqa: E402

# Tried in this order; the first one that answers is used. (The old .../downloads/activefires/activefires.csv
# link returns 404, so the feed now comes from the CWFIS web feature service as CSV.)
FEED_URLS = [u for u in [
    os.environ.get("ACTIVE_FIRES_URL"),
    "https://geoserver.cwfif.nrcan.gc.ca/geoserver/wfs?service=WFS&version=2.0.1&request=GetFeature&outputFormat=csv"
    "&typeName=public:cwfif_national_activefires&sortBy=agency_code+A,record_start+D"
    "&CQL_FILTER=now()%3E=record_start%20AND%20now()%3C=record_end",
    "https://cwfis.cfs.nrcan.gc.ca/geoserver/public/ows?service=WFS&version=1.0.0&request=GetFeature"
    "&typeName=public:activefires_current&outputFormat=csv",
] if u]
LOG_PATH = os.path.join(HERE, "data", "scored_log.csv")
RUN_PATH = os.path.join(HERE, "data", "latest_run.json")
MODEL_DIR = os.path.join(ROOT, "models")
MODEL_FILE, INFO_FILE = "final_model_v14.7_allyears.pkl", "final_model_v14.7_info.json"
BIG_HA = 100.0
CLOSED_STAGES = {"UC", "EX", "OUT"}       # under control / extinguished (OC = out of control and BH = being held stay open)

LOG_COLS = ["fire_id", "agency", "name", "report_date", "first_seen_utc", "last_seen_utc", "lat", "lon",
            "scored", "raw_score", "chance_big", "alert", "threshold", "ha_first", "ha_max", "stage_last",
            "status", "big_so_far", "warnings"]

# canonical name -> possible column names in the feed (compared in lower case)
COLS = {
    "lat": ["lat", "latitude"],
    "lon": ["lon", "long", "longitude"],
    "start": ["startdate", "start_date", "fire_start_date", "firedate", "ignition_date", "reported_date"],  # optional
    "ha": ["fire_size", "hectares", "current_size", "size_ha", "area_ha", "ha", "size", "area"],
    "agency": ["agency_code", "agency", "src_agency", "source_agency"],
    "fid": ["national_fire_id"],
    "name": ["agency_fire_id", "firename", "fire_name", "name", "fire_number", "fire_num", "fire_id", "id"],
    "stage": ["stage_of_control_status", "stage_of_control", "stageofcontrol", "stage", "fire_status", "status"],
    "prescribed": ["fire_was_prescribed", "prescribed"],
}
REQUIRED = ["lat", "lon"]


# ------------------------------------------------------------------ feed
def normalise_feed(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename feed columns to lat, lon, start, ha, agency, name, stage. Fails loudly if lat/lon/start are missing."""
    lower = {c.lower().strip(): c for c in raw.columns}
    pick = {}
    for canon, options in COLS.items():
        for o in options:
            if o in lower:
                pick[canon] = lower[o]; break
    if "lat" not in pick or "lon" not in pick:          # some feeds only carry a WKT point: POINT(lon lat)
        g = next((lower[k] for k in ("geom", "the_geom", "geometry", "shape") if k in lower), None)
        if g is not None:
            xy = raw[g].astype(str).str.extract(r"POINT\s*\(\s*(-?[\d.]+)\s+(-?[\d.]+)")
            raw = raw.assign(_lon=pd.to_numeric(xy[0], errors="coerce"), _lat=pd.to_numeric(xy[1], errors="coerce"))
            lower = {c.lower().strip(): c for c in raw.columns}
            pick["lon"], pick["lat"] = "_lon", "_lat"
    missing = [c for c in REQUIRED if c not in pick]
    if missing:
        raise KeyError(f"Active-fire list has no column for {missing}. Columns found: {list(raw.columns)}. "
                       f"Add the right names to COLS in automation/daily_score.py.")
    out = pd.DataFrame({k: raw[v] for k, v in pick.items()})
    for k in ("ha", "agency", "name", "stage", "fid", "start", "prescribed"):
        if k not in out: out[k] = np.nan
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out["ha"] = pd.to_numeric(out["ha"], errors="coerce")
    out["start"] = pd.to_datetime(out["start"], errors="coerce")
    out["agency"] = out["agency"].fillna("").astype(str).str.strip().str.upper()
    out["name"] = out["name"].fillna("").astype(str).str.strip()
    out["stage"] = out["stage"].map(_stage_code)
    out["prescribed"] = out["prescribed"].astype(str).str.strip().str.lower().isin(["true", "1", "y", "yes", "t"])
    out = out.dropna(subset=["lat", "lon"])
    out = out[out["lat"].between(41, 84) & out["lon"].between(-142, -52)]
    fid = out["fid"].fillna("").astype(str).str.strip()
    fallback = out["agency"] + "|" + out["name"] + "|" + out["lat"].round(2).astype(str) + "|" + out["lon"].round(2).astype(str)
    out["fire_id"] = np.where(fid.ne("") & fid.ne("nan"), fid, fallback)
    return out.drop_duplicates("fire_id").reset_index(drop=True)


def _stage_code(v) -> str:
    """Turn the stage text into OC / BH / UC / OUT (accepts codes or words)."""
    t = str(v).strip().upper()
    if t in ("", "NAN", "NONE"): return ""
    if t.startswith("OUT OF") or t == "OC": return "OC"
    if t.startswith("BEING") or t == "BH": return "BH"
    if t.startswith("UNDER") or t == "UC": return "UC"
    if t.startswith("OUT") or t in ("EX", "EXTINGUISHED"): return "OUT"
    return t


def fetch_raw_feed(http_get=requests.get) -> tuple:
    """Return (raw table, url used). Tries each address in FEED_URLS until one gives a table."""
    from io import StringIO
    problems = []
    for url in FEED_URLS:
        try:
            r = http_get(url, timeout=90)
            r.raise_for_status()
            raw = pd.read_csv(StringIO(r.text))
            if len(raw.columns) < 3:
                raise ValueError(f"reply is not a table: {r.text[:120]!r}")
            return raw, url
        except Exception as e:  # noqa: BLE001
            problems.append(f"{url[:70]}... -> {type(e).__name__}: {str(e)[:120]}")
    raise RuntimeError("No active-fire address worked: " + " || ".join(problems))


def fetch_feed(http_get=requests.get) -> pd.DataFrame:
    return normalise_feed(fetch_raw_feed(http_get)[0])


# --------------------------------------------------------------- features
def fetch_weather(lat, lon, report_date, http_get=requests.get) -> dict:
    """Open-Meteo: the 7 days strictly before the report date (same recipe as training)."""
    params = {"latitude": lat, "longitude": lon, "past_days": 10, "forecast_days": 1, "timezone": "auto",
              "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max",
                        "wind_gusts_10m_max", "relative_humidity_2m_mean", "sunshine_duration"]}
    r = http_get("https://api.open-meteo.com/v1/forecast", params=params, timeout=30)
    r.raise_for_status()
    d = r.json()["daily"]
    days = pd.to_datetime(d["time"])
    keep = (days >= report_date - pd.Timedelta(days=7)) & (days < report_date)
    if keep.sum() < 5:
        raise ValueError(f"only {int(keep.sum())} weather days available before {report_date.date()}")
    a = {k: np.array(v, dtype=float)[keep] for k, v in d.items() if k != "time"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "temperature_2m_max_mean": np.nanmean(a["temperature_2m_max"]),
            "temperature_2m_max_max": np.nanmax(a["temperature_2m_max"]),
            "temperature_2m_max_min": np.nanmin(a["temperature_2m_max"]),
            "temperature_2m_min_mean": np.nanmean(a["temperature_2m_min"]),
            "precipitation_sum_sum": np.nansum(a["precipitation_sum"]),
            "precipitation_sum_mean": np.nanmean(a["precipitation_sum"]),
            "wind_speed_10m_max_mean": np.nanmean(a["wind_speed_10m_max"]),
            "wind_speed_10m_max_max": np.nanmax(a["wind_speed_10m_max"]),
            "wind_gusts_10m_max_mean": np.nanmean(a["wind_gusts_10m_max"]),
            "relative_humidity_2m_mean_mean": np.nanmean(a["relative_humidity_2m_mean"]),
            "relative_humidity_2m_mean_min": np.nanmin(a["relative_humidity_2m_mean"]),
            "relative_humidity_2m_mean_max": np.nanmax(a["relative_humidity_2m_mean"]),
            "sunshine_duration_mean": np.nanmean(a["sunshine_duration"]),
        }


class EarthEngineTerrain:
    """Elevation, slope and NDVI, same sources as data_pipeline/03_fetch_terrain.py."""
    def __init__(self):
        import ee
        key = os.environ.get("EE_SERVICE_ACCOUNT_JSON")
        if not key:
            raise RuntimeError("Secret EE_SERVICE_ACCOUNT_JSON is not set (needed for NDVI, elevation and slope).")
        info = json.loads(key)
        ee.Initialize(ee.ServiceAccountCredentials(info["client_email"], key_data=key), project=info.get("project_id"))
        self.ee = ee
        self.srtm = ee.Image("USGS/SRTMGL1_003")
        self.slope_srtm = ee.Terrain.slope(self.srtm)
        self.dem_fb = ee.ImageCollection("COPERNICUS/DEM/GLO30").mosaic().select("DEM")   # above 60N
        self.slope_fb = ee.Terrain.slope(self.dem_fb)

    def __call__(self, lat, lon, report_date) -> dict:
        ee = self.ee
        pt = ee.Geometry.Point([lon, lat])
        mean = ee.Reducer.mean()
        elev = self.srtm.reduceRegion(mean, pt, 30).get("elevation").getInfo()
        if elev is None:        # keep BOTH fallbacks together (see the note in 03_fetch_terrain.py)
            elev = self.dem_fb.reduceRegion(mean, pt, 30).get("DEM").getInfo()
            slope = self.slope_fb.reduceRegion(mean, pt, 30).get("slope").getInfo()
        else:
            slope = self.slope_srtm.reduceRegion(mean, pt, 30).get("slope").getInfo()
        start = (report_date - pd.Timedelta(days=16)).strftime("%Y-%m-%d")
        img = ee.ImageCollection("MODIS/061/MOD13Q1").filterDate(start, report_date.strftime("%Y-%m-%d")).select("NDVI").mosaic()
        v = img.reduceRegion(mean, pt, 250).get("NDVI").getInfo()
        return {"elevation": elev, "slope": slope, "NDVI": (v / 10000.0) if v is not None else np.nan}


# ---------------------------------------------------------------- scoring
class Model:
    def __init__(self, model_dir=MODEL_DIR):
        import joblib
        self.model = joblib.load(os.path.join(model_dir, MODEL_FILE))
        info = json.load(open(os.path.join(model_dir, INFO_FILE)))
        self.features = info["features"]
        self.threshold = float(info["threshold_recall"])
        self.cal = info["calibrator"]

    def score(self, row: dict) -> tuple:
        X = pd.DataFrame([{f: row.get(f, np.nan) for f in self.features}], columns=self.features).astype(float)
        raw = float(self.model.predict_proba(X)[:, 1][0])
        p = float(np.clip(raw, 1e-4, 1 - 1e-4))
        chance = 1.0 / (1.0 + np.exp(-(self.cal["coef"] * np.log(p / (1 - p)) + self.cal["intercept"])))
        return raw, float(chance), raw >= self.threshold


def build_features(fire, deps, rd) -> tuple:
    """Return (feature dict, list of warnings). rd = report date. Raises if weather or terrain cannot be fetched."""
    warns = []
    rd = pd.Timestamp(rd).normalize()
    row = {"LATITUDE": fire["lat"], "LONGITUDE": fire["lon"],
           "province_encoded": PROVINCE_MAP.get(fire["agency"], np.nan)}
    if fire["agency"] not in PROVINCE_MAP:
        warns.append(f"agency {fire['agency']!r} not in province map; province left empty")
    row.update(deps["weather"](fire["lat"], fire["lon"], rd))
    row.update(deps["terrain"](fire["lat"], fire["lon"], rd))
    try:
        row.update(deps["lookup"].get(fire["lat"], fire["lon"]))
    except ValueError as e:
        warns.append(f"road/population lookup failed ({e}); left empty")
    if not np.isfinite(row.get("NDVI", np.nan)):
        warns.append("NDVI not available yet; left empty")
    return row, warns


# -------------------------------------------------------------------- run
def run(feed: pd.DataFrame, log: pd.DataFrame, deps: dict, now=None) -> tuple:
    """Pure function: returns (new log, stats). deps = weather, terrain, lookup, model.

    The first run (empty log) only records the fires already burning. After that, a fire that is not in the log
    is new: today is its report date, it is scored once, and never re-scored.
    """
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    stamp = now.strftime("%Y-%m-%d %H:%M")
    today = now.normalize()
    first_run = len(log) == 0
    log = log.copy()
    for c in LOG_COLS:
        if c not in log: log[c] = np.nan
    stats = {"time_utc": stamp, "feed_fires": int(len(feed)), "baseline": 0, "new_scored": 0, "new_failed": 0,
             "skipped_prescribed": 0, "updated": 0, "closed": 0, "errors": []}

    # 1) update fires already in the log
    seen = feed.set_index("fire_id")
    known = log["fire_id"].isin(seen.index)
    for i in log.index[known]:
        f = seen.loc[log.at[i, "fire_id"]]
        ha = f["ha"]
        if np.isfinite(ha):
            log.at[i, "ha_max"] = np.nanmax([log.at[i, "ha_max"], ha])
        log.at[i, "stage_last"] = f["stage"]
        log.at[i, "last_seen_utc"] = stamp
        log.at[i, "status"] = "closed" if f["stage"] in CLOSED_STAGES else "open"
        stats["updated"] += 1
    gone = (~known) & (log["status"] != "closed") & log["fire_id"].notna()
    log.loc[gone, "status"] = "closed"                # no longer on the active list
    stats["closed"] = int(gone.sum())

    # 2) new fires
    new = feed[~feed["fire_id"].isin(log["fire_id"])]
    rows = []
    for _, fire in new.iterrows():
        base = {"fire_id": fire["fire_id"], "agency": fire["agency"], "name": fire["name"],
                "report_date": today.strftime("%Y-%m-%d"), "first_seen_utc": stamp, "last_seen_utc": stamp,
                "lat": fire["lat"], "lon": fire["lon"], "ha_first": fire["ha"], "ha_max": fire["ha"],
                "stage_last": fire["stage"], "status": "closed" if fire["stage"] in CLOSED_STAGES else "open",
                "big_so_far": False}
        if first_run or fire["prescribed"]:
            why = "already burning when tracking started" if first_run else "prescribed burn"
            rows.append({**base, "scored": False, "alert": False, "warnings": f"not scored: {why}"})
            stats["baseline" if first_run else "skipped_prescribed"] += 1
            continue
        try:
            feats, warns = build_features(fire, deps, today)
            raw, chance, alert = deps["model"].score(feats)
        except Exception as e:  # noqa: BLE001
            stats["new_failed"] += 1; stats["errors"].append(f"{fire['fire_id']}: {e}")
            continue
        rows.append({**base, "scored": True, "raw_score": round(raw, 4), "chance_big": round(chance, 4),
                     "alert": bool(alert), "threshold": round(deps["model"].threshold, 4),
                     "warnings": "; ".join(warns)})
        stats["new_scored"] += 1
    if rows:
        log = pd.concat([log, pd.DataFrame(rows)], ignore_index=True) if len(log) else pd.DataFrame(rows)
    for c in LOG_COLS:
        if c not in log: log[c] = np.nan
    log["big_so_far"] = pd.to_numeric(log["ha_max"], errors="coerce") > BIG_HA
    return log[LOG_COLS], stats


def _annotate(level, title, msg):
    """GitHub shows these lines as annotations on the run page (and the API can read them)."""
    print(f"::{level} title={title}::{str(msg).replace(chr(10), ' | ')[:900]}")


def main():
    try:
        _main()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        _annotate("error", "Daily scoring failed", f"{type(e).__name__}: {e}")
        sys.exit(1)


def _main():
    raw, url = fetch_raw_feed()
    _annotate("notice", "Feed columns", f"{len(raw)} rows from {url[:60]}...; columns={list(raw.columns)}")
    print(raw.head(3).to_string())
    feed = normalise_feed(raw)
    log = pd.read_csv(LOG_PATH) if os.path.exists(LOG_PATH) else pd.DataFrame(columns=LOG_COLS)
    if len(log) == 0 or (~feed["fire_id"].isin(log["fire_id"])).sum() == 0:
        deps = {}                     # nothing to score: do not start Earth Engine or load the model
    else:
        deps = {"weather": fetch_weather, "terrain": EarthEngineTerrain(), "model": Model()}
        from road_pop_lookup import RoadPopLookup
        deps["lookup"] = RoadPopLookup()
    log, stats = run(feed, log, deps)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log.to_csv(LOG_PATH, index=False)
    json.dump(stats, open(RUN_PATH, "w"), indent=1)
    print(json.dumps(stats, indent=1))
    _annotate("notice", "Run summary", f"feed={stats['feed_fires']} new_scored={stats['new_scored']} "
              f"new_failed={stats['new_failed']} baseline={stats['baseline']} updated={stats['updated']} "
              f"closed={stats['closed']}")
    for e in stats["errors"][:3]:
        _annotate("warning", "Fire not scored", e)
    if stats["new_failed"] and not stats["new_scored"]:
        _annotate("error", "Every new fire failed", stats["errors"][0] if stats["errors"] else "")
        sys.exit(1)


if __name__ == "__main__":
    main()
