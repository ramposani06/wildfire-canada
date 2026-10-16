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
EMPTY = lambda: pd.DataFrame(columns=ds.LOG_COLS)  # noqa: E731


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


def raw_feed(extra=None):
    """Looks like the real CWFIS feed: no start date column."""
    d = pd.DataFrame({
        "agency_code": ["bc", "ab", "bc"], "national_fire_id": ["BC-1", "AB-2", "BC-3"],
        "agency_fire_id": ["K1", "A2", "OLD"], "latitude": [56.0, 50.0, 52.0], "longitude": [-120.0, -114.0, -121.0],
        "fire_size": [5.0, 150.0, 3.0], "stage_of_control_status": ["OC", "Being Held", "UC"],
        "fire_was_prescribed": ["false", "false", "false"]})
    return d if extra is None else pd.concat([d, extra], ignore_index=True)


def new_fire(**kw):
    row = {"agency_code": "sk", "national_fire_id": "SK-9", "agency_fire_id": "S9", "latitude": 57.0, "longitude": -105.0,
           "fire_size": 1.0, "stage_of_control_status": "OC", "fire_was_prescribed": "false"}
    row.update(kw)
    return pd.DataFrame([row])


def test_real_feed_columns_are_understood():
    f = ds.normalise_feed(raw_feed())
    assert f["fire_id"].tolist() == ["BC-1", "AB-2", "BC-3"]
    assert f["agency"].tolist() == ["BC", "AB", "BC"] and f["stage"].tolist() == ["OC", "BH", "UC"]
    assert f["ha"].tolist() == [5.0, 150.0, 3.0]


def test_missing_core_column_fails_loudly():
    with pytest.raises(KeyError, match="Columns found"):
        ds.normalise_feed(raw_feed().drop(columns=["latitude"]))


def test_first_run_only_records_fires_already_burning():
    log, st = ds.run(ds.normalise_feed(raw_feed()), EMPTY(), deps(), NOW)
    assert st["baseline"] == 3 and st["new_scored"] == 0
    assert (log["scored"] == False).all() and log["raw_score"].isna().all()  # noqa: E712


def test_a_fire_that_appears_later_is_scored_once_with_today_as_report_date():
    feed = ds.normalise_feed(raw_feed())
    log, _ = ds.run(feed, EMPTY(), deps(), NOW)
    feed2 = ds.normalise_feed(raw_feed(new_fire()))
    log2, st2 = ds.run(feed2, log, deps(), NOW + pd.Timedelta(hours=12))
    assert st2["new_scored"] == 1
    r = log2[log2["fire_id"] == "SK-9"].iloc[0]
    assert bool(r["scored"]) and bool(r["alert"]) and r["raw_score"] == 0.8 and r["report_date"] == "2026-07-11"
    log3, st3 = ds.run(feed2, log2, deps(), NOW + pd.Timedelta(days=1))          # third run: nothing new
    assert st3["new_scored"] == 0 and len(log3) == len(log2)


def test_growth_is_tracked_and_big_flag_set():
    feed = ds.normalise_feed(raw_feed())
    log, _ = ds.run(feed, EMPTY(), deps(), NOW)
    feed2 = feed.copy(); feed2.loc[feed2["fire_id"] == "BC-1", "ha"] = 400.0
    log2, _ = ds.run(feed2, log, deps(), NOW + pd.Timedelta(days=1))
    r = log2[log2["fire_id"] == "BC-1"].iloc[0]
    assert r["ha_max"] == 400.0 and bool(r["big_so_far"]) and r["ha_first"] == 5.0


def test_fire_leaving_the_list_or_under_control_is_closed():
    feed = ds.normalise_feed(raw_feed())
    log, _ = ds.run(feed, EMPTY(), deps(), NOW)
    assert log[log["fire_id"] == "BC-3"].iloc[0]["status"] == "closed"            # stage UC
    log2, st = ds.run(feed[feed["fire_id"] == "BC-3"], log, deps(), NOW + pd.Timedelta(days=1))
    assert (log2["status"] == "closed").all() and st["closed"] >= 1


def test_failed_fires_are_not_logged_so_they_retry_next_run():
    feed = ds.normalise_feed(raw_feed())
    log, _ = ds.run(feed, EMPTY(), deps(), NOW)
    feed2 = ds.normalise_feed(raw_feed(new_fire()))
    log2, st = ds.run(feed2, log, deps(fail_weather=True), NOW + pd.Timedelta(hours=12))
    assert st["new_failed"] == 1 and len(log2) == len(log) and st["errors"]
    log3, st3 = ds.run(feed2, log2, deps(), NOW + pd.Timedelta(hours=24))
    assert st3["new_scored"] == 1


def test_prescribed_burns_are_not_scored():
    feed = ds.normalise_feed(raw_feed())
    log, _ = ds.run(feed, EMPTY(), deps(), NOW)
    feed2 = ds.normalise_feed(raw_feed(new_fire(fire_was_prescribed="true")))
    log2, st = ds.run(feed2, log, deps(), NOW + pd.Timedelta(hours=12))
    assert st["skipped_prescribed"] == 1 and st["new_scored"] == 0


def test_unknown_agency_and_lookup_failure_only_warn():
    feed = ds.normalise_feed(raw_feed())
    log, _ = ds.run(feed, EMPTY(), deps(), NOW)
    feed2 = ds.normalise_feed(raw_feed(new_fire(agency_code="zz", latitude=69.5)))
    log2, st = ds.run(feed2, log, deps(), NOW + pd.Timedelta(hours=12))
    w = log2[log2["fire_id"] == "SK-9"].iloc[0]["warnings"]
    assert st["new_scored"] == 1 and "province" in w and "lookup" in w


def test_stage_words_become_codes():
    assert [ds._stage_code(x) for x in ["Out of Control", "Being Held", "Under Control", "Out", "OC", ""]] == \
           ["OC", "BH", "UC", "OUT", "OC", ""]


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


def test_feed_with_only_a_wkt_point_column():
    raw = pd.DataFrame({"agency_code": ["bc"], "fire_name": ["W1"], "fire_size": [3.0], "geom": ["POINT (-120.5 56.25)"]})
    f = ds.normalise_feed(raw)
    assert f.iloc[0]["lat"] == 56.25 and f.iloc[0]["lon"] == -120.5 and f.iloc[0]["agency"] == "BC"


def test_feed_falls_back_to_the_next_address():
    calls = []
    class R:
        def __init__(self, ok): self.ok, self.text = ok, "a,b,c\n1,2,3\n"
        def raise_for_status(self):
            if not self.ok: raise RuntimeError("404")
    def get(url, timeout=0):
        calls.append(url); return R(len(calls) > 1)
    old = ds.FEED_URLS
    ds.FEED_URLS = ["http://one", "http://two"]
    try:
        raw, used = ds.fetch_raw_feed(get)
    finally:
        ds.FEED_URLS = old
    assert used == "http://two" and list(raw.columns) == ["a", "b", "c"]
