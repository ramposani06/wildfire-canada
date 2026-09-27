"""
Train the production big-fire risk model: LightGBM + Platt calibration,
on the frozen temporal split (train 2012-2021, calibrate 2022-2024).

The alert threshold is chosen by maximizing F1 on the CALIBRATION set
only. It is never re-tuned on 2025/2026 — those years exist solely to
measure real-world performance, not to pick knobs.
"""
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from feature_config import (
    FEATURE_COLS_V14, MEAN_IMPUTE_COLS, ZERO_IMPUTE_COLS,
    TRAIN_YEARS, CALIBRATION_YEARS,
)


def logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def build_X(df: pd.DataFrame, fill_means: dict, cols=FEATURE_COLS_V14) -> pd.DataFrame:
    X = df[cols].copy()
    for c, m in fill_means.items():
        if c in X.columns:
            X[c] = X[c].fillna(m)
    for c in ZERO_IMPUTE_COLS:
        if c in X.columns:
            X[c] = X[c].fillna(0)
    return X


def train_final_model(train_table: pd.DataFrame, out_path: str = "models/final_model.pkl"):
    train_df = train_table[train_table.year.between(*TRAIN_YEARS)].copy()
    calib_df = train_table[train_table.year.between(*CALIBRATION_YEARS)].copy()

    fill_means = {c: train_df[c].mean() for c in MEAN_IMPUTE_COLS}

    model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.05,
        is_unbalance=True, random_state=42, verbose=-1,
    )
    model.fit(build_X(train_df, fill_means), train_df.is_big_fire)

    raw_calib = model.predict_proba(build_X(calib_df, fill_means))[:, 1]
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(
        logit(raw_calib).reshape(-1, 1), calib_df.is_big_fire,
    )
    calib_scores = platt.predict_proba(logit(raw_calib).reshape(-1, 1))[:, 1]

    # Pick the alert threshold that maximizes F1 on the calibration set
    # ONLY. Never re-tuned against 2025/2026.
    best_thresh, best_f1 = 0.5, -1
    for t in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(calib_df.is_big_fire, calib_scores >= t)
        if f1 > best_f1:
            best_thresh, best_f1 = t, f1

    bundle = {
        "model": model,
        "platt_calibrator": platt,
        "feature_cols": FEATURE_COLS_V14,
        "fill_means": fill_means,
        "alert_threshold": best_thresh,
        "version": "v14.1",
    }
    joblib.dump(bundle, out_path)
    print(f"Saved {out_path} — alert threshold {best_thresh:.2f} (F1={best_f1:.3f} on calibration set)")
    return bundle


def score(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    """Apply a trained bundle to new fire records, returning calibrated probabilities."""
    X = build_X(df, bundle["fill_means"], bundle["feature_cols"])
    raw = bundle["model"].predict_proba(X)[:, 1]
    return bundle["platt_calibrator"].predict_proba(logit(raw).reshape(-1, 1))[:, 1]


if __name__ == "__main__":
    train_table = pd.read_csv("train_table.csv")
    train_final_model(train_table)
