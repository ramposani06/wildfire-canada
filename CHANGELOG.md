# Changelog

This project went through many iterations before landing on the current production model. Earlier work wasn't tracked in git — it happened across a few Colab notebooks — so this changelog reconstructs the real version history from those notebooks (kept in `notebooks/exploratory/`) rather than pretending development started from a clean slate.

Every version below is a real, distinct model that was actually trained and saved during this project. Rejected ideas are listed too — the rejections are as much a part of the record as the adoptions.

## v3 / v4 — first models

Random Forest baseline classifiers, first pass at features (weather + a first version of satellite pre-detection). `model_v3_fixed_satellite.pkl`, `model_v4_rf.pkl`.

## v6 — correctly-bounded satellite features

Fixed the satellite pre-detection window (MODIS/VIIRS counts and max fire radiative power in the 7 days *before* the official report — an earlier version of this window let post-report detections leak in). Random Forest, oversampled with `RandomOverSampler`. `model_v6_bounded_satellite_BEST.pkl`.

## v7 — added province

Added `province_encoded` to v6's feature set. Notably, adding province changed elevation's feature importance meaningfully — a sign that some of what "elevation" was capturing was really regional variation. `model_v7_with_province_BEST.pkl`.

## v10 — switched to LightGBM

Replaced Random Forest with LightGBM, and switched to a **temporal split** (train on earlier years, test on later years) instead of a random split — the first step toward the forward-testing discipline the rest of the project follows. `model_v10_lightgbm_BEST.pkl`.

## v11 — calibration added (experimental, not adopted as production)

Added Platt and isotonic calibration on top of v10, with reliability tables and calibration plots comparing predicted probability to real big-fire rate. Saved separately as `calibrated_model_v11.pkl` — did not yet replace v10 as production.

## v12 — first frozen production model

Trained 2012–2021, calibrated (Platt) on 2022–2024, alert threshold chosen by best F1 on the calibration set only. FWI (Fire Weather Index, via CFFDRS) was tested in earlier exploratory work (`notebooks/exploratory/01_early_fwi_and_weather_fetch.ipynb`) and **dropped** — it didn't earn its place in the final feature set. Also fixed a known data issue (Alberta 2023 satellite data). 2025 was explicitly excluded from training to keep it available as a forward test. `final_model_v12.pkl`.

## Rejected — hyperparameter tuning (Optuna, 80 trials)

Tuned LightGBM hyperparameters against 2022–2024 as the selection metric. The tuned model looked better on that same selection metric — expected, since it was optimized against it — but this was flagged as likely optimistic before ever being trusted, and it did not survive a genuine forward-test comparison against the frozen model's simple, untuned defaults. Rejected; the frozen model kept its original hyperparameters.

## v13 — road distance

Added `dist_to_road_m` (nearest-road distance via Statistics Canada's National Road Network), a genuine accessibility signal. Missing values (1,613 fires, mostly Parks Canada) imputed with the training-set median. This was the first added feature after v12 that showed a real, reproducible gain under bootstrap testing. `final_model_v13.pkl`.

## Rejected — nearby recent-fire count

Tested whether the count of *other* fires within a radius in the days before a report (a regional-outbreak signal — lightning storms or extreme weather often produce many fires at once) improved on v13. Checked year-by-year across 2022–2024. Not adopted — the population-density experiment below showed a stronger, more consistent signal for the same modeling slot.

## v14 — population density (current-generation production model)

Added `pop_within_10km` and `pop_within_25km` (Kontur Population Dataset) on top of v13. Adopted on evidence across 4 of 5 tested periods, with 2025 showing a null result rather than a negative one — consistent with the project's standard of requiring a real, not just lucky, effect. `final_model_v14.pkl`.

## v14.1 — NT/YT slope data-quality fix

While investigating consistently weaker performance in the Northwest Territories and Yukon, found that **all 3,016 NT/YT fires in the training data had wrong slope values** (near-zero, mean 0.18°) because the terrain-fetch pipeline's fallback for locations above SRTM's ~60°N coverage limit was wired up for elevation but never for slope. Fresh Earth Engine queries against real fire coordinates confirmed the correct values (mean slope 5.37° after the fix). Retrained and validated with bootstrap confidence intervals across every province and both forward-test years before being adopted — no province showed a confirmed regression, and 2026 showed a confirmed national improvement. `final_model_v14_1.pkl` — current production model.

## v14.2 to v14.6 — pre-2012 years, leak clean-up, recovered 2006-07, calibration (Sept 2026)

Scripts for this work are in `analysis/` and `audits/`.

- **Feature check (`analysis/threshold_and_features.py`, `compare_v14_1_vs_v14_4.py`):** the saved v14.4 model used 40 features, including rejected ones. Compared with v14.1 on the same fires.
- **Leak found:** `n_modis_matches` and `has_modis_match` count satellite detections over the whole fire, which leaks the final fire size. They are excluded for good, along with `YEAR_clean`. v14.4 is not to be used.
- **v14.5 (`analysis/retrain_v14_5_clean.py`):** honest clean retrain with feature-group selection on 2022-2024 (nothing extra adopted).
- **Recovered 2006-2007 (`analysis/v14_6/recover_2006_2007.py`):** 16,613 fires were missing from the unified dataset. Recovered from the historical files with a self-test (known rows rebuilt first, written only if 99%+ match).
- **Sensor flag (`analysis/v14_6/check_sensor_flag.py`):** `sensor_available` and `sat_zero` give no real gain; `sat_zero` is stale. Not used. VIIRS is exactly 0 before 2012.
- **v14.6 (`analysis/v14_6/retrain_v14_6_clean.py`):** 24 features, trained 2004-2021. Training from 2004 beat training from 2012 (validation PR-AUC 0.600 vs 0.587). Forward test 2025+: ROC-AUC 0.929, PR-AUC 0.634. Alert threshold 0.770 (raw score) or top 15%.
- **QA audit (`audits/qa_audit_v14_6.py`):** 14 of 14 checks passed. Saved model reproduces reported scores, bootstrap ranges, province/cause breakdowns, alert stability, sensor eras (leave-one-year-out), direction checks.
- **Data audit (`audits/data_integrity_audit.py`):** no impossible values, no year-to-year jumps, recovered years look normal, NDVI taken 1-16 days before the fire. Small notes: 10 bad road distances, empty UNIQUE_ID for 2012+.
- **Calibration (`analysis/v14_6/calibration_v14_6.py`):** Platt calibrator fitted on 2022-2024 (from a model trained to 2021). On 2025+ Brier 0.114 -> 0.061, slope 1.00.

Model files (`final_model_v14.6_clean*.pkl`, `final_model_v14.6_calibrator.pkl`) and the training data stay on Google Drive.

## Also tested and rejected (see model card for full list)

- FBP fuel type
- Terrain ruggedness (near-random signal, AUC 0.529)
- Distance to water (0.899 correlated with road distance — no independent signal)
