"""Load and apply the v14.6 calibrator (raw score -> estimated chance of a big fire).

The calibrator is stored as plain JSON (numbers only) by analysis/v14_6/calibration_v14_6.py,
so it loads anywhere without pickle or the original script.

It is for DISPLAY only. Alerts use the raw score (info["threshold_recall"] = 0.770).
It was fitted on 2022-2024 and can drift in a very different fire season; refit when new years arrive.
"""
import json

import numpy as np


def _logit(p, eps):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def load_calibrator(path: str) -> dict:
    with open(path) as f:
        cal = json.load(f)
    if cal.get("method") not in ("platt", "isotonic"):
        raise ValueError(f"Unknown calibrator method: {cal.get('method')!r}")
    return cal


def apply_calibrator(cal: dict, raw) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    eps = cal.get("eps", 1e-4)
    if cal["method"] == "platt":
        z = cal["coef"] * _logit(raw, eps) + cal["intercept"]
        return 1.0 / (1.0 + np.exp(-z))
    return np.clip(np.interp(raw, cal["x"], cal["y"]), eps, 1 - eps)
