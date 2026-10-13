"""Tests for automation/daily_score.py with fake web data and a stand-in model (no network, no LightGBM).
Run: pip install pytest && pytest tests -q"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "automation"))
import daily_score as ds  # noqa: E402

NOW = pd.Timestamp("2026-07-10 15:00")


class StubModel:
    threshold = 0.5
    def score(self, row):
        raw = 0.8 if row["LATITUDE"] > 55 else 0.2
        return raw, raw / 2, raw >= self.threshold


class StubLookup:
    def get(self, lat, lon):
        if lat > 69: raise ValueError("outside")
        return {"dist_to_road_m": 500.0, "pop_within_10km": 10.0, "pop_within_25km": 40.0}


def deps(fail_weather=False):
    def weather(lat, lon, rd):
        if fail_weather: raise RuntimeError("weather down")
        return {"temperature_2m_max_mean": 20.0}
    return {"weather": weather, "terrain": lambda lat, lon, rd: {"elevation": 300.0, "slope": 2.0, "NDVI": 0.6},
            "lookup": StubLookup(), "model": StubModel()}


def raw_feed():
    return pd.DataFrame({
        "AGENCY": ["bc", "ab", "sk", "bc"], "FIRENAME": ["K1", "A2", "S3", "OLD"],
        "LAT": [56.0, 50.0, 51.0, 52.0], "LON": [-120.0, -114.0, -105.0, -121.0],
        "STARTDATE": ["2026-07-10", "2026-07-09", "2026-06-01", "2026-07-10"],
        "HECTARES": [5.0, 150.0, 20.0, 3.0], "STAGE_OF_CONTROL": ["OC", "BH", "OC", "UC"]})


def test_feed_columns_are_found_case_insensitively():
    f = ds.normalise_feed(raw_feed())
    assert {"lat", "lon", "start", "ha", "agency", "name", "stage", "fire_id"} <= set(f.columns)
    assert f["agency"].tolist() == ["BC", "AB", "SK", "BC"]


def test_missing_core_column_fails_loudly():
    with pytest.raises(KeyError, match="Columns found"):
        ds.normalise_feed(raw_feed().drop(columns=["LAT"]))


def test_only_new_fires_are_scored_and_old_ones_skipped():
    log, st = ds.run(ds.normalise_feed(raw_feed()), pd.DataFrame(columns=ds.LOG_COLS), deps(), NOW)
    assert st["new_scored"] == 3 and st["new_failed"] == 0      # SK fire started in June -> not "new"
    assert set(log["name"]) == {"K1", "A2", "OLD"}
    k1 = log[log["name"] == "K1"].iloc[0]
    assert bool(k1["alert"]) and k1["raw_score"] == 0.8


def test_second_run_does_not_duplicate_and_tracks_growth():
    feed = ds.normalise_feed(raw_feed())
    log, _ = ds.run(feed, pd.DataFrame(columns=ds.LOG_COLS), deps(), NOW)
    feed2 = feed.copy(); feed2.loc[feed2["name"] == "K1", "ha"] = 400.0
    log2, st2 = ds.run(feed2, log, deps(), NOW + pd.Timedelta(days=1))
    assert len(log2) == len(log) and st2["new_scored"] == 0
    k1 = log2[log2["name"] == "K1"].iloc[0]
    assert k1["ha_max"] == 400.0 and bool(k1["big_so_far"]) and k1["ha_first"] == 5.0


def test_fire_leaving_the_list_or_under_control_is_closed():
    feed = ds.normalise_feed(raw_feed())
    log, _ = ds.run(feed, pd.DataFrame(columns=ds.LOG_COLS), deps(), NOW)
    assert log[log["name"] == "OLD"].iloc[0]["status"] == "closed"            # stage UC
    log2, st = ds.run(feed[feed["name"] == "OLD"], log, deps(), NOW + pd.Timedelta(days=1))
    assert (log2["status"] == "closed").all() and st["closed"] >= 1


def test_failed_fires_are_not_logged_so_they_retry_next_run():
    log, st = ds.run(ds.normalise_feed(raw_feed()), pd.DataFrame(columns=ds.LOG_COLS), deps(fail_weather=True), NOW)
    assert st["new_failed"] == 3 and len(log) == 0 and st["errors"]


def test_unknown_agency_and_lookup_failure_only_warn():
    feed = ds.normalise_feed(raw_feed().iloc[:1].assign(AGENCY="zz", LAT=69.5))
    log, st = ds.run(feed, pd.DataFrame(columns=ds.LOG_COLS), deps(), NOW)
    assert st["new_scored"] == 1
    assert "province" in log.iloc[0]["warnings"] and "lookup" in log.iloc[0]["warnings"]


def test_weather_uses_the_seven_days_before_the_report_date():
    days = [f"2026-07-{d:02d}" for d in range(1, 11)]
    daily = {"time": days}
    for k in ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max",
              "wind_gusts_10m_max", "relative_humidity_2m_mean", "sunshine_duration"]:
        daily[k] = list(range(1, 11))                      # value = day number
    class R:
        def raise_for_status(self): pass
        def json(self): return {"daily": daily}
    w = ds.fetch_weather(50, -100, pd.Timestamp("2026-07-10"), http_get=lambda *a, **k: R())
    assert w["temperature_2m_max_max"] == 9 and w["temperature_2m_max_min"] == 3     # days 3..9
    assert w["precipitation_sum_sum"] == sum(range(3, 10))
