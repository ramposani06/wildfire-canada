"""
Score a brand-new fire report in near real time with the v14.6 model.

What changed from the v14.1 scorer:
  - Uses the v14.6 files: raw-score model + info json + separate calibrator.
  - Alert rule is on the RAW score (0.770), the same rule that was tested.
    The calibrated "chance" is for display only.
  - Satellite features now match training: MODIS and VIIRS detections within
    10 km, from 7 days before through the report day (both ends included).
    The old scorer used 1 day, a 0.5 degree box and VIIRS = 0.
  - A failed satellite request gives EMPTY values (the model handles them),
    never a fake "0 detections", and the result says so.
  - Bad inputs (0,0 coordinates, road distance over 1,000 km) are caught.

Files needed (see docs/model_card.md):
  final_model_v14.6_clean_allyears.pkl   model for live use
  final_model_v14.6_clean_info.json      feature list and thresholds
  final_model_v14.6_calibrator.json      raw score -> estimated chance (plain numbers)

Environment: FIRMS_API_KEY (never put keys in code or notebooks).
"""
import importlib.util
import json
import os
import sys
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "modeling"))
from feature_config import PROVINCE_MAP  # noqa: E402
from calibration import apply_calibrator, load_calibrator  # noqa: E402

FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
SATELLITE_SOURCES = {"modis": "MODIS_NRT", "viirs": "VIIRS_SNPP_NRT"}
WINDOW_DAYS = 7          # 7 days before the report day, plus the report day
RADIUS_KM = 10
MAX_DAY_RANGE = 5        # FIRMS area API limit per request
MAX_ROAD_M = 1_000_000   # anything larger is a failed lookup (10 such rows in the training data)
CANADA_BOX = (41.0, 84.0, -142.0, -52.0)   # lat_min, lat_max, lon_min, lon_max


# ---------------------------------------------------------------- inputs
def check_location(lat: float, lon: float) -> None:
    lat_min, lat_max, lon_min, lon_max = CANADA_BOX
    if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
        raise ValueError(f"Location ({lat}, {lon}) is outside Canada. Check for a (0,0) or swapped coordinates.")


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------- weather
def fetch_live_weather(lat: float, lon: float, http_get=requests.get) -> dict:
    """Open-Meteo forecast API with past_days: the archive API lags by a few
    days. The 7 days BEFORE today are used (today itself is dropped), which
    matches training (7 days strictly before the report date)."""
    params = {
        "latitude": lat, "longitude": lon, "past_days": 7, "forecast_days": 1,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum",
                  "wind_speed_10m_max", "wind_gusts_10m_max",
                  "relative_humidity_2m_mean", "sunshine_duration"],
        "timezone": "auto",
    }
    r = http_get("https://api.open-meteo.com/v1/forecast", params=params, timeout=30)
    r.raise_for_status()
    d = {k: np.array(v, dtype=float)[:-1] for k, v in r.json()["daily"].items() if k != "time"}
    return {
        "temperature_2m_max_mean": np.nanmean(d["temperature_2m_max"]),
        "temperature_2m_max_max": np.nanmax(d["temperature_2m_max"]),
        "temperature_2m_max_min": np.nanmin(d["temperature_2m_max"]),
        "temperature_2m_min_mean": np.nanmean(d["temperature_2m_min"]),
        "precipitation_sum_sum": np.nansum(d["precipitation_sum"]),
        "precipitation_sum_mean": np.nanmean(d["precipitation_sum"]),
        "wind_speed_10m_max_mean": np.nanmean(d["wind_speed_10m_max"]),
        "wind_speed_10m_max_max": np.nanmax(d["wind_speed_10m_max"]),
        "wind_gusts_10m_max_mean": np.nanmean(d["wind_gusts_10m_max"]),
        "relative_humidity_2m_mean_mean": np.nanmean(d["relative_humidity_2m_mean"]),
        "relative_humidity_2m_mean_min": np.nanmin(d["relative_humidity_2m_mean"]),
        "relative_humidity_2m_mean_max": np.nanmax(d["relative_humidity_2m_mean"]),
        "sunshine_duration_mean": np.nanmean(d["sunshine_duration"]),
    }


# --------------------------------------------------------------- terrain
def _load_terrain_module():
    """data_pipeline/03_fetch_terrain.py starts with a digit, so it cannot be
    imported by name. Load it from its file path."""
    path = os.path.join(HERE, "..", "data_pipeline", "03_fetch_terrain.py")
    spec = importlib.util.spec_from_file_location("fetch_terrain", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fetch_live_terrain(lat: float, lon: float, report_date: pd.Timestamp) -> dict:
    """Same Earth Engine lookups as training (elevation, slope, NDVI from the
    16 days up to the report date). Needs Earth Engine sign-in."""
    mod = _load_terrain_module()
    t = mod.fetch_terrain_one(lat, lon)
    return {**t, "NDVI": mod.fetch_ndvi_one(lat, lon, report_date)}


# ------------------------------------------------------------- satellite
def _bbox(lat: float, lon: float, radius_km: float) -> str:
    dlat = radius_km / 111.0 * 1.2                                   # 20% margin, exact filter comes later
    dlon = radius_km / (111.0 * max(np.cos(np.radians(lat)), 0.05)) * 1.2
    return f"{lon - dlon:.4f},{lat - dlat:.4f},{lon + dlon:.4f},{lat + dlat:.4f}"


def _fetch_source(source: str, lat: float, lon: float, start: pd.Timestamp, end: pd.Timestamp,
                  api_key: str, http_get) -> pd.DataFrame:
    """All detections for one source from start to end (inclusive), in chunks of at most 5 days."""
    frames, day = [], start
    while day <= end:
        n = min(MAX_DAY_RANGE, (end - day).days + 1)
        url = f"{FIRMS_BASE_URL}/{api_key}/{source}/{_bbox(lat, lon, RADIUS_KM)}/{n}/{day:%Y-%m-%d}"
        r = http_get(url, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"FIRMS {source} request failed with status {r.status_code}")
        text = r.text.strip()
        if text and text.lower().startswith("latitude"):
            frames.append(pd.read_csv(pd.io.common.StringIO(text)))
        elif text:
            raise RuntimeError(f"FIRMS {source} returned an unexpected reply: {text[:80]!r}")
        # an empty reply with status 200 is a real "no detections" answer
        day += timedelta(days=n)
    if not frames:
        return pd.DataFrame(columns=["latitude", "longitude", "acq_date", "frp"])
    return pd.concat(frames, ignore_index=True).drop_duplicates()   # guards against overlapping chunks


def fetch_live_satellite(lat: float, lon: float, report_date: pd.Timestamp, api_key: str,
                         http_get=requests.get) -> tuple[dict, list]:
    """MODIS and VIIRS detections within 10 km, 7 days before through the report day.
    Returns (features, warnings). A failed request gives empty (NaN) values, not zeros."""
    report_date = pd.Timestamp(report_date).normalize()
    start = report_date - pd.Timedelta(days=WINDOW_DAYS)
    out, warns = {}, []
    for name, source in SATELLITE_SOURCES.items():
        try:
            det = _fetch_source(source, lat, lon, start, report_date, api_key, http_get)
        except Exception as e:  # noqa: BLE001
            out[f"{name}_count_early7d"] = np.nan
            out[f"{name}_max_frp_early7d"] = np.nan
            warns.append(f"{name} satellite data unavailable ({e}); left empty")
            continue
        if len(det):
            dist = haversine_km(lat, lon, det["latitude"].astype(float), det["longitude"].astype(float))
            when = pd.to_datetime(det["acq_date"])
            det = det[(dist <= RADIUS_KM) & (when >= start) & (when <= report_date)]
        out[f"{name}_count_early7d"] = float(len(det))
        out[f"{name}_max_frp_early7d"] = float(det["frp"].max()) if len(det) else 0.0
    return out, warns


# ---------------------------------------------------------------- scorer
class LiveScorer:
    """Scores one newly reported fire with the frozen v14.6 model."""

    def __init__(self, model_dir: str = "models",
                 model_file: str = "final_model_v14.6_clean_allyears.pkl",
                 info_file: str = "final_model_v14.6_clean_info.json",
                 calibrator_file: str = "final_model_v14.6_calibrator.json"):
        self.model = joblib.load(os.path.join(model_dir, model_file))
        self.info = json.load(open(os.path.join(model_dir, info_file)))
        self.features = self.info["features"]
        self.threshold = float(self.info["threshold_recall"])       # 0.770, on the raw score
        cal_path = os.path.join(model_dir, calibrator_file)
        self.calibrator = load_calibrator(cal_path) if os.path.exists(cal_path) else None

    def build_row(self, row: dict) -> pd.DataFrame:
        missing = [f for f in self.features if f not in row]
        if missing:
            raise KeyError(f"Missing features: {missing}")
        return pd.DataFrame([{f: row[f] for f in self.features}], columns=self.features).astype(float)

    def score_row(self, row: dict, warnings: list | None = None) -> dict:
        """Score an already-built feature row (used by score_fire and by tests)."""
        X = self.build_row(row)
        raw = float(self.model.predict_proba(X)[:, 1][0])
        chance = float(apply_calibrator(self.calibrator, [raw])[0]) if self.calibrator is not None else None
        return {
            "raw_score": raw,
            "estimated_chance_big_fire": chance,      # calibrated on 2022-2024; can drift, display only
            "alert": raw >= self.threshold,           # the tested alert rule
            "alert_threshold_raw": self.threshold,
            "warnings": warnings or [],
        }

    def score_fire(self, lat: float, lon: float, province: str, report_date: pd.Timestamp,
                   dist_to_road_m: float, pop_within_10km: float, pop_within_25km: float,
                   firms_api_key: str, http_get=requests.get) -> dict:
        check_location(lat, lon)
        if province not in PROVINCE_MAP:
            raise ValueError(f"Unknown province {province!r}; use one of {sorted(PROVINCE_MAP)}")
        report_date = pd.Timestamp(report_date)
        if abs((pd.Timestamp.now().normalize() - report_date.normalize()).days) > 1:
            raise ValueError("Live scoring is for fires reported today or yesterday (weather comes from the forecast API).")
        warns = []
        if dist_to_road_m is None or dist_to_road_m > MAX_ROAD_M or dist_to_road_m < 0:
            warns.append("road distance looked wrong and was left empty")
            dist_to_road_m = np.nan

        weather = fetch_live_weather(lat, lon, http_get)
        terrain = fetch_live_terrain(lat, lon, report_date)
        satellite, sat_warns = fetch_live_satellite(lat, lon, report_date, firms_api_key, http_get)
        warns += sat_warns

        row = {**weather, **terrain, **satellite,
               "dist_to_road_m": dist_to_road_m,
               "pop_within_10km": pop_within_10km, "pop_within_25km": pop_within_25km,
               "province_encoded": PROVINCE_MAP[province]}
        return self.score_row(row, warns)


if __name__ == "__main__":
    scorer = LiveScorer()
    print(scorer.score_fire(
        lat=55.2, lon=-118.8, province="AB", report_date=pd.Timestamp.now(),
        dist_to_road_m=1200.0, pop_within_10km=350, pop_within_25km=2100,
        firms_api_key=os.environ["FIRMS_API_KEY"],
    ))
