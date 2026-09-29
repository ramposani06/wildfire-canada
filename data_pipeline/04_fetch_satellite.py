"""
Match NASA FIRMS MODIS/VIIRS satellite fire detections to reported fires,
to build a "was this fire already visible from space before it was
officially reported" feature set.

Two lessons learned the hard way:
  - Never cache a failed/empty API request as if it were a confirmed
    "no detections" result — a transient error and a genuine null result
    look the same in a naive cache and are NOT the same thing. Only cache
    successful responses.
  - Window: detections within 10 km, from 7 days BEFORE the report date
    through the report day itself (both ends included). Detections after the
    report day are never counted (leakage guard).
"""
import os
import time

import pandas as pd
import requests

FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
CHUNK_CACHE_DIR = "sat_chunks"
WINDOW_DAYS = 7
MATCH_RADIUS_KM = 10


def get_chunk(api_key: str, source: str, date: str, region_bbox: str) -> pd.DataFrame:
    """
    Fetch (and cache) one day's worth of detections for one source
    (MODIS_NRT / VIIRS_SNPP_NRT / etc.) over a bounding box.
    A failed request is NOT cached — only a successful one is.
    """
    os.makedirs(CHUNK_CACHE_DIR, exist_ok=True)
    cache_path = f"{CHUNK_CACHE_DIR}/{source}_{date}.csv"
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    url = f"{FIRMS_BASE_URL}/{api_key}/{source}/{region_bbox}/1/{date}"
    r = requests.get(url, timeout=60)
    if r.status_code != 200 or not r.text.strip():
        # Do NOT cache this — a transient failure must be retried later,
        # not silently treated as "zero detections".
        return pd.DataFrame()

    df = pd.read_csv(pd.io.common.StringIO(r.text))
    df.to_csv(cache_path, index=False)
    return df


def fetch_source_range(api_key: str, source: str, dates: list[str], region_bbox: str) -> pd.DataFrame:
    chunks = []
    for date in dates:
        chunk = get_chunk(api_key, source, date, region_bbox)
        if not chunk.empty:
            chunks.append(chunk)
        time.sleep(0.5)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def match_detections_to_fires(fires: pd.DataFrame, detections: pd.DataFrame,
                               window_days: int = WINDOW_DAYS,
                               radius_km: float = MATCH_RADIUS_KM) -> pd.DataFrame:
    """
    For each fire, count satellite detections within radius_km and within
    the k-days-before-report window, where 0 <= k <= window_days (k = 0 is the
    report day itself). Detections AFTER the report date are excluded (leakage guard).
    """
    from sklearn.neighbors import BallTree
    import numpy as np

    det_rad = np.radians(detections[["latitude", "longitude"]].values)
    tree = BallTree(det_rad, metric="haversine")

    results = []
    for fire in fires.itertuples():
        fire_rad = np.radians([[fire.LATITUDE, fire.LONGITUDE]])
        idx = tree.query_radius(fire_rad, r=radius_km / 6371.0)[0]
        nearby = detections.iloc[idx]

        nearby_dates = pd.to_datetime(nearby["acq_date"])
        days_before = (fire.REP_DATE - nearby_dates).dt.days
        valid = nearby[(days_before >= 0) & (days_before <= window_days)]

        results.append({
            "UNIQUE_ID": fire.UNIQUE_ID,
            "count_early7d": len(valid),
            "max_frp_early7d": valid["frp"].max() if len(valid) else 0.0,
        })
    return pd.DataFrame(results)


def build_satellite_features(fires: pd.DataFrame, api_key: str, region_bbox: str) -> pd.DataFrame:
    dates = pd.date_range(
        fires.REP_DATE.min() - pd.Timedelta(days=WINDOW_DAYS + 1),
        fires.REP_DATE.max(),
    ).strftime("%Y-%m-%d").tolist()

    modis = fetch_source_range(api_key, "MODIS_NRT", dates, region_bbox)
    viirs = fetch_source_range(api_key, "VIIRS_SNPP_NRT", dates, region_bbox)

    modis_feats = match_detections_to_fires(fires, modis) if not modis.empty else pd.DataFrame()
    viirs_feats = match_detections_to_fires(fires, viirs) if not viirs.empty else pd.DataFrame()

    out = fires[["UNIQUE_ID"]].copy()
    out = out.merge(modis_feats.rename(columns={
        "count_early7d": "modis_count_early7d", "max_frp_early7d": "modis_max_frp_early7d"
    }), on="UNIQUE_ID", how="left")
    out = out.merge(viirs_feats.rename(columns={
        "count_early7d": "viirs_count_early7d", "max_frp_early7d": "viirs_max_frp_early7d"
    }), on="UNIQUE_ID", how="left")

    for c in ["modis_count_early7d", "viirs_count_early7d"]:
        out[c] = out[c].fillna(0)
    for c in ["modis_max_frp_early7d", "viirs_max_frp_early7d"]:
        out[c] = out[c].fillna(0.0)

    return out


if __name__ == "__main__":
    fires = pd.read_csv("fires_to_fetch.csv", parse_dates=["REP_DATE"])
    api_key = os.environ["FIRMS_API_KEY"]
    result = build_satellite_features(fires, api_key, region_bbox="-141,41,-52,84")
    result.to_csv("satellite_output.csv", index=False)
    print(f"Fetched satellite features for {len(result):,} fires")
