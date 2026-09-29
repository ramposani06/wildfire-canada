"""Tests for live_scoring/score_live_fire.py using fake web replies and a small stand-in model.
Run: pip install pytest && pytest tests -q"""
import json
import os
import sys
import types

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "live_scoring"))
import score_live_fire as live  # noqa: E402

FEATS = json.load(open(os.path.join(ROOT, "tests", "features_v14_6.json")))


class FakeResp:
    def __init__(self, text="", status=200, js=None):
        self.text, self.status_code, self._js = text, status, js

    def json(self):
        return self._js

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError("bad status")


def firms_csv(rows):
    head = "latitude,longitude,acq_date,frp\n"
    return head + "".join(f"{a},{b},{c},{d}\n" for a, b, c, d in rows)


@pytest.fixture()
def scorer(tmp_path):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(600, len(FEATS))), columns=FEATS)
    y = (X.iloc[:, 0] + rng.normal(size=600) > 1).astype(int)
    model = HistGradientBoostingClassifier(max_iter=30).fit(X, y)
    joblib.dump(model, tmp_path / "final_model_v14.6_clean_allyears.pkl")
    json.dump({"features": FEATS, "threshold_recall": 0.770}, open(tmp_path / "final_model_v14.6_clean_info.json", "w"))

    json.dump({"method": "platt", "coef": 1.0, "intercept": -1.0, "eps": 1e-4},
              open(tmp_path / "final_model_v14.6_calibrator.json", "w"))
    return live.LiveScorer(model_dir=str(tmp_path))


def test_feature_order_and_alert_rule(scorer):
    row = {f: 0.0 for f in FEATS}
    out = scorer.score_row(row)
    assert set(out) >= {"raw_score", "estimated_chance_big_fire", "alert", "alert_threshold_raw", "warnings"}
    assert out["alert"] == (out["raw_score"] >= 0.770)          # alert uses the RAW score
    r = min(max(out["raw_score"], 1e-4), 1 - 1e-4)
    assert out["estimated_chance_big_fire"] == pytest.approx(1 / (1 + np.exp(-(np.log(r / (1 - r)) - 1.0))))


def test_missing_feature_is_an_error(scorer):
    with pytest.raises(KeyError):
        scorer.score_row({"NDVI": 0.5})


def test_empty_values_are_allowed(scorer):
    row = {f: np.nan for f in FEATS}
    assert 0 <= scorer.score_row(row)["raw_score"] <= 1


def test_location_check():
    live.check_location(55.2, -118.8)
    for bad in [(0, 0), (55.2, 118.8), (10, -100)]:
        with pytest.raises(ValueError):
            live.check_location(*bad)


def test_satellite_window_radius_and_chunks():
    lat, lon = 55.0, -118.0
    report = pd.Timestamp("2026-07-10")
    calls = []

    def http_get(url, timeout=60):
        calls.append(url)
        if "MODIS" in url:
            rows = [
                (55.02, -118.0, "2026-07-10", 30.0),   # inside: report day, ~2 km
                (55.02, -118.0, "2026-07-03", 12.0),   # inside: exactly 7 days before
                (55.02, -118.0, "2026-07-02", 99.0),   # too old (8 days before)
                (55.02, -118.0, "2026-07-11", 99.0),   # after the report day
                (55.30, -118.0, "2026-07-08", 99.0),   # ~33 km away
            ]
            return FakeResp(firms_csv(rows))
        return FakeResp("")                              # VIIRS: real "nothing found"

    out, warns = live.fetch_live_satellite(lat, lon, report, "KEY", http_get)
    assert out["modis_count_early7d"] == 2 and out["modis_max_frp_early7d"] == 30.0
    assert out["viirs_count_early7d"] == 0 and out["viirs_max_frp_early7d"] == 0.0
    assert warns == []
    modis_calls = [u for u in calls if "MODIS" in u]
    assert len(modis_calls) == 2                          # 8 days -> chunks of 5 + 3
    assert modis_calls[0].endswith("/5/2026-07-03") and modis_calls[1].endswith("/3/2026-07-08")


def test_failed_satellite_request_is_empty_not_zero():
    def http_get(url, timeout=60):
        return FakeResp("Invalid MAP_KEY", status=200) if "VIIRS" in url else FakeResp("", status=500)

    out, warns = live.fetch_live_satellite(55.0, -118.0, pd.Timestamp("2026-07-10"), "KEY", http_get)
    for k in ("modis_count_early7d", "modis_max_frp_early7d", "viirs_count_early7d", "viirs_max_frp_early7d"):
        assert np.isnan(out[k])
    assert len(warns) == 2


def test_weather_uses_seven_days_before_today():
    days = list(range(8))                                  # 7 past days + today
    daily = {"time": [f"d{i}" for i in days],
             "temperature_2m_max": [10.0 + i for i in days], "temperature_2m_min": [1.0] * 8,
             "precipitation_sum": [0.0, 1, 2, 3, 4, 5, 6, 100.0], "wind_speed_10m_max": [5.0] * 8,
             "wind_gusts_10m_max": [9.0] * 8, "relative_humidity_2m_mean": [50.0] * 8,
             "sunshine_duration": [3600.0] * 8}
    w = live.fetch_live_weather(55.0, -118.0, lambda *a, **k: FakeResp(js={"daily": daily}))
    assert w["precipitation_sum_sum"] == 21.0              # today's 100 is dropped
    assert w["temperature_2m_max_max"] == 16.0
    assert len(w) == 13


def test_full_flow_flags_bad_road_distance(scorer, monkeypatch):
    days = list(range(8))
    daily = {"time": [str(i) for i in days], "temperature_2m_max": [20.0] * 8, "temperature_2m_min": [8.0] * 8,
             "precipitation_sum": [0.0] * 8, "wind_speed_10m_max": [15.0] * 8, "wind_gusts_10m_max": [30.0] * 8,
             "relative_humidity_2m_mean": [40.0] * 8, "sunshine_duration": [40000.0] * 8}

    def http_get(url, params=None, timeout=30):
        return FakeResp(js={"daily": daily}) if "open-meteo" in url else FakeResp("")

    fake_mod = types.SimpleNamespace(fetch_terrain_one=lambda la, lo: {"elevation": 800.0, "slope": 4.0},
                                     fetch_ndvi_one=lambda la, lo, d: 0.6)
    monkeypatch.setattr(live, "_load_terrain_module", lambda: fake_mod)
    out = scorer.score_fire(55.2, -118.8, "AB", pd.Timestamp.now(), dist_to_road_m=9_000_000,
                            pop_within_10km=350, pop_within_25km=2100, firms_api_key="KEY", http_get=http_get)
    assert any("road distance" in w for w in out["warnings"])
    assert 0 <= out["raw_score"] <= 1
    with pytest.raises(ValueError):
        scorer.score_fire(55.2, -118.8, "XX", pd.Timestamp.now(), 1200, 1, 1, "KEY", http_get)


def test_isotonic_calibrator_roundtrip():
    sys.path.insert(0, os.path.join(ROOT, "modeling"))
    from calibration import apply_calibrator
    cal = {"method": "isotonic", "x": [0.0, 0.5, 1.0], "y": [0.0, 0.2, 0.9], "eps": 1e-4}
    out = apply_calibrator(cal, [0.25, 0.75, 1.0])
    assert out[0] == pytest.approx(0.1) and out[1] == pytest.approx(0.55) and out[2] == pytest.approx(0.9)
