"""
Fetch 7-day pre-report weather for a list of fires from the Open-Meteo
Archive API.

Key lessons baked into this script (learned the hard way during this
project):
  - Batching too many locations into one multi-location request triggers
    Open-Meteo's rate limiter more aggressively than the same number of
    requests spread out. Cap batch size at MAX_BATCH.
  - Always reindex to a declared column list before appending to CSV —
    dict-order drift between rows silently corrupts the header/column
    alignment otherwise.
  - Checkpoint incrementally (append-only) so a crash or rate-limit pause
    never loses completed work; resume by diffing against what's already
    written.
"""
import os
import time

import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm

MAX_BATCH = 30
BACKOFF_BASE_SECONDS = 60
REQUEST_PACING_SECONDS = 1.0

DAILY_VARS = [
    "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
    "wind_speed_10m_max", "wind_gusts_10m_max",
    "relative_humidity_2m_mean", "sunshine_duration",
]

DECLARED_COLS = [
    "UNIQUE_ID",
    "temperature_2m_max_mean", "temperature_2m_max_max", "temperature_2m_max_min",
    "temperature_2m_min_mean",
    "precipitation_sum_sum", "precipitation_sum_mean",
    "wind_speed_10m_max_mean", "wind_speed_10m_max_max",
    "wind_gusts_10m_max_mean",
    "relative_humidity_2m_mean_mean", "relative_humidity_2m_mean_min",
    "relative_humidity_2m_mean_max",
    "sunshine_duration_mean",
]


def fetch_batch_with_retry(fires_group: pd.DataFrame, rep_date, max_retries: int = 4):
    """Fetch weather for one date's worth of fires (<= MAX_BATCH rows)."""
    start_date = (pd.Timestamp(rep_date) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = (pd.Timestamp(rep_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    lats = ",".join(fires_group.LATITUDE.astype(str))
    lons = ",".join(fires_group.LONGITUDE.astype(str))

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lats, "longitude": lons,
        "start_date": start_date, "end_date": end_date,
        "daily": DAILY_VARS, "timezone": "auto",
    }

    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 429:
                time.sleep(BACKOFF_BASE_SECONDS * (attempt + 1))
                continue
            if r.status_code != 200:
                return [], f"status {r.status_code}"

            data = r.json()
            results = data if isinstance(data, list) else [data]
            if len(results) != len(fires_group):
                return [], "count mismatch"

            rows = []
            for fire, res in zip(fires_group.itertuples(), results):
                dist = ((fire.LATITUDE - res.get("latitude", 999)) ** 2
                        + (fire.LONGITUDE - res.get("longitude", 999)) ** 2) ** 0.5
                d = res.get("daily", {})
                if dist > 0.5 or not d.get("temperature_2m_max"):
                    continue
                rows.append({
                    "UNIQUE_ID": fire.UNIQUE_ID,
                    "temperature_2m_max_mean": np.mean(d["temperature_2m_max"]),
                    "temperature_2m_max_max": np.max(d["temperature_2m_max"]),
                    "temperature_2m_max_min": np.min(d["temperature_2m_max"]),
                    "temperature_2m_min_mean": np.mean(d["temperature_2m_min"]),
                    "precipitation_sum_sum": np.sum(d["precipitation_sum"]),
                    "precipitation_sum_mean": np.mean(d["precipitation_sum"]),
                    "wind_speed_10m_max_mean": np.mean(d["wind_speed_10m_max"]),
                    "wind_speed_10m_max_max": np.max(d["wind_speed_10m_max"]),
                    "wind_gusts_10m_max_mean": np.mean(d["wind_gusts_10m_max"]),
                    "relative_humidity_2m_mean_mean": np.mean(d["relative_humidity_2m_mean"]),
                    "relative_humidity_2m_mean_min": np.min(d["relative_humidity_2m_mean"]),
                    "relative_humidity_2m_mean_max": np.max(d["relative_humidity_2m_mean"]),
                    "sunshine_duration_mean": np.mean(d["sunshine_duration"]),
                })
            return rows, None
        except Exception:
            time.sleep(5)
    return [], "failed all retries"


def fetch_weather(fires: pd.DataFrame, out_path: str):
    """
    fires must have columns: UNIQUE_ID, LATITUDE, LONGITUDE, REP_DATE.
    Resumable: re-running only fetches fires not already in out_path.
    """
    if not os.path.exists(out_path):
        pd.DataFrame(columns=DECLARED_COLS).to_csv(out_path, index=False)

    already_done = set(pd.read_csv(out_path).UNIQUE_ID)
    todo = fires[~fires.UNIQUE_ID.isin(already_done)]
    print(f"Already done: {len(already_done):,}, remaining: {len(todo):,}")

    sub_batches = []
    for rep_date, group in todo.groupby(todo.REP_DATE.dt.date):
        for i in range(0, len(group), MAX_BATCH):
            sub_batches.append((rep_date, group.iloc[i:i + MAX_BATCH]))
    print(f"Sub-batches (max {MAX_BATCH} fires each): {len(sub_batches)}")

    errors = []
    for rep_date, group in tqdm(sub_batches, desc="Weather (capped batches)"):
        rows, err = fetch_batch_with_retry(group, rep_date)
        if rows:
            pd.DataFrame(rows)[DECLARED_COLS].to_csv(out_path, mode="a", header=False, index=False)
        if err:
            errors.append((rep_date, err))
        time.sleep(REQUEST_PACING_SECONDS)

    final = pd.read_csv(out_path)
    print(f"\nTotal fires with weather now: {len(final):,} of {len(fires):,}")
    print(f"Sub-batches with errors: {len(errors)}")
    return final


if __name__ == "__main__":
    fires = pd.read_csv("fires_to_fetch.csv", parse_dates=["REP_DATE"])
    fetch_weather(fires, out_path="weather_output.csv")
