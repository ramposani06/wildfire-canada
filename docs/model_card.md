# Model card - Canada wildfire "big fire" model, v14.6

Date: 2026-10-01 (rerun on repaired data v4: scores, alert rules, calibration, ablation, FWI baseline; sections still on v3 are marked)

## Main model (decision 2026-10-01)
**The main model is the no-satellite model (20 features).** It uses only information known before the report day: ROC-AUC 0.901 (0.893-0.909), PR-AUC 0.531 (0.506-0.562) on 2025+. Alert threshold on its raw score: **0.688** (not 0.770). The full 24-feature model (0.923 / 0.621) is kept as an **optional satellite-enhanced model**, to be used only when satellite detections are timestamped and known to be earlier than the scoring time. Reason: the whole satellite gain comes from report-day detections whose timing relative to the report is unknown (see Satellite timing audit). The sections below that describe "the model" and the 0.770 rule refer to the full 24-feature model unless they say otherwise.

## What it does
Predicts which Canadian wildfires will become big fires (more than 100 ha), using satellite, weather, terrain, road and population data for each fire. It uses only information from before or on the report day.

## Files (in the wildfire_project folder on Drive)
| File | Use |
|---|---|
| `final_model_v14.6_nosat_allyears.pkl` + `final_model_v14.6_nosat_info.json` | **Main model for live use** (no satellite, trained on all years, threshold 0.688). Drive only. |
| `final_model_v14.6_clean_allyears.pkl` | Optional satellite-enhanced model (24 features, trained on all years, threshold 0.770) |
| `final_model_v14.6_clean.pkl` | Same model trained through 2024. Gave the test scores below. Use for reports. |
| `final_model_v14.6_clean_info.json` | Feature list, threshold, top-15% setting |
| `final_model_v14.6_calibrator.json` | Turns the raw score into a real chance (see Calibration). Plain numbers, no pickle. |
| `unified_dataset_2004_2026_FINAL_v4.csv` | Training data (v3 plus 2,584 repaired 2025 fires, see Data repair) |

## Data
- 142,990 fires, 2004-2026 (v4). The 2006 and 2007 fires (16,613) were missing from v2 and are now recovered. v3 had 140,406; see Data repair below.
- 152 fires with impossible coordinates (mostly 0,0) are left out of training and testing (142,838 rows used).
- Big fires are about 7.7% of all fires, and 11.5% in 2025+ (11,145 test fires).
- The 2026 fires were built separately from the live CWFIS feed (per project notes): the report date is the first day a fire appeared in the feed, so it is approximate, and small fires still burning were left out. That makes the 2026 big-fire rate (13.5%) a little high.

## Data repair (2026-10-01)
- **Problem:** v3 had holes in 2025 (for example Manitoba and Ontario showed 0 fires). Cause: an inner join on the weather table silently dropped fires that had no weather row.
- **Fix:** weather and terrain were rebuilt for the missing 2025 fires with the same pipeline code. A check on 150 known fires gave 100% identical weather; terrain was 99% identical (2,604 of 2,610). 2,584 fires were added (v3 140,406 to v4 142,990). 22 fires with impossible report dates were skipped.
- **Still missing:** Yukon 2022 (289 fires, no report date), about 15 fires without weather, about 59 without road distance.
- **Effect on the score (2025+):**

| | ROC-AUC v3 | ROC-AUC v4 | PR-AUC v3 | PR-AUC v4 |
|---|---|---|---|---|
| 2025+ | 0.929 | 0.923 | 0.634 | 0.621 |
| 2025 only | 0.923 | 0.915 | 0.587 | 0.582 |
| 2026 only | 0.929 | 0.929 | 0.656 | 0.656 |

  The 0.770 alert rule on 2025+ moves from 73% caught / 53% precise to 72% / 52%. The drop is small. The 2025 test grew from 3,603 to 6,184 fires, 2025+ from 8,564 to 11,145. Calibration still holds (2025: 10.4% predicted vs 9.9% actual).
- **Which numbers are which (2026-10-01):** scores, alert rules (0.770 and top 15%), calibration, feature-group ablation, FWI baseline, and the 2004-2011 zone test were rerun on **v4**. Still on **v3** and marked as such: alert rules by year, by-province table, within-province gains, QA audit, data audit.

## Model
- LightGBM classifier: 200 trees, learning rate 0.05, max depth 8, 31 leaves, class weighting on (`is_unbalance`).
- Because of the class weighting, **raw scores are not real probabilities** (average raw score 0.27 vs 11% actual big fires). Do not show raw scores as percentages. To show a chance, use the calibrator (see Calibration).
- Older versions (v12 to v14) were calibrated and used an alert threshold near 0.28-0.29. **The v14.6 model file gives raw scores: use 0.770 (or the top 15%) on them, not 0.28.**

## The 24 features
- **Weather (13):** temperature_2m_max (mean, max, min), temperature_2m_min_mean, precipitation_sum (sum, mean), wind_speed_10m_max (mean, max), wind_gusts_10m_max_mean, relative_humidity_2m_mean (mean, min, max), sunshine_duration_mean. Window: the 7 days before the report, strictly before the report date.
- **Land (4):** NDVI, elevation, slope, province_encoded
- **Satellite (4):** modis_count_early7d, modis_max_frp_early7d, viirs_count_early7d, viirs_max_frp_early7d. Window: detections within 10 km, from 7 days before through the report day.
- **Access and people (3):** dist_to_road_m, pop_within_10km, pop_within_25km

### Timing of each feature
| Feature | Known at report time? | Note |
|---|---|---|
| Weather (13) | Yes | 7 days before the report, strictly before |
| Satellite (4) | Yes | 7 days before through the report day, 10 km |
| Elevation, slope, province | Yes | Fixed for a location |
| dist_to_road_m | Yes | Uses today's road map for older fires. Tested earlier: small effect. |
| NDVI | Yes (checked) | Taken 1 to 16 days before the fire (median 9). None on or after the fire date. |
| pop_within_10km / 25km | **To confirm** | Not written down whether the population layer is per year or one modern layer. |

## How it was tested
- Train up to 2021. Choose features and threshold on 2022-2024. Score 2025+ at the end.
- 2025+ is a **forward-in-time test, not a perfectly untouched one**: earlier model versions were looked at on these years too. The first truly untouched test will be 2027.
- Training on 2004 onward beat training on 2012 onward (validation PR-AUC 0.600 vs 0.587).

## Scores (v4, repaired data)
| Data | Fires | ROC-AUC | PR-AUC |
|---|---|---|---|
| **Test 2025+** | 11,145 (11.5% big) | **0.923** (0.916-0.930) | **0.621** (0.593-0.650) |
| 2025 only | 6,184 | 0.915 | 0.582 |
| 2026 only | 4,961 | 0.929 | 0.656 |
| Validation 2022-2024 | 18,010 | 0.909 | 0.599 |

The 2025+ ranges are 95% bootstrap intervals (500 resamples) on v4; the by-year rows have no ranges yet. On v3 the 2025 and 2026 ranges overlapped, so the gap between those years may be noise. Random guessing would give a PR-AUC of about 0.115.

| Model (same 2025+ fires, v4) | ROC-AUC | PR-AUC |
|---|---|---|
| v14.6 | 0.923 | 0.621 |
| v14.1 (saved file) | 0.916 | 0.596 |
| v14.4 (has leaky columns, do not use) | 0.884 | 0.529 |

v14.5 was not rerun.

## Alert rules (on 2025+)
| Rule | Big fires caught | Flagged fires that are truly big | Share of fires flagged |
|---|---|---|---|
| Score at or above 0.770 | 72% | 52% | 16% |
| Top 15% by score (cut 0.788) | 69% | 53% | 15% |

**By year (v3, not rerun):**
| | 0.770 rule: flagged / caught / precision | Top 15% rule: caught / precision |
|---|---|---|
| 2025 | 11% / 65% / 49% | 76% / 42% |
| 2026 | 19% / 76% / 55% | 67% / 61% |

With the fixed 0.770 rule, the share flagged follows how bad the season is (11% in a milder year, 19% in a bad one). That is expected, not a fault. The top-15% rule always flags 15%, but recall and precision then move by year.

**Top-risk fires (v4, 2025+):** the top 5% of fires by score contain 32% of the big fires, the top 10% contain 54%, the top 15% contain 69%, the top 20% contain 80%, and the top 25% contain 87%.

## Calibration (rerun on v4, 2026-10-01)
The calibrator (Platt, a simple curve) was fitted on 2022-2024 scores from a model trained only to 2021, then tested on 2025+. It does not change the ranking, so ROC-AUC and PR-AUC stay the same.

| 2025+ | Average score | Actual rate | Brier (lower is better) | Slope (1.0 = perfect) |
|---|---|---|---|---|
| Raw score | 0.277 | 0.115 | 0.119 | 0.88 |
| Calibrated | 0.120 | 0.115 | **0.063** | **1.00** |
| Guessing the overall rate | | | 0.102 | |

- 2025: calibrated average 0.104 vs actual 0.099. 2026: 0.140 vs 0.135. The calibrator held across both very different seasons.
- Reliability bins: the biggest gap between predicted and actual in any group was 2.4 points; the average gap was 0.5 points.
- **What the numbers mean:** raw score 0.770 is about a 29% chance; the top 5% of fires are about 63% or more; the median fire is about 2%.
- The saved 2024-trained model with the same calibrator also worked (average 0.121 vs actual 0.115).
- **Limits:** the calibrator was fitted on 2022-24 and can drift in a very different season. It was fitted for the 2024-trained model's kind of scores, so for the all-years model it is a close approximation, not a tested match. The calibrator file was not refit on v4 (2022-2024 data did not change), and the v4 check above still passes. Refit it when 2027 fires are in. Small provinces and rare groups (for example human-caused fires) were not checked one by one.
- Use the calibrated chance only for showing a risk number. The alert rules (0.770 raw score, or top 15%) stay as they are.

## What drives the score, and how it compares with the fire weather index (2026-09-29)

**Feature groups (v4; refit on 2004-2024, tested on 2025-2026, 11,150 fires, 11.5% big).** Each group was removed in turn, then used alone.

| Setup | Features | ROC-AUC | PR-AUC | Big fires in top 15% |
|---|---|---|---|---|
| All features | 24 | 0.922 | 0.620 | 70% |
| Without weather | 11 | 0.917 | 0.610 | 69% |
| Without terrain | 21 | 0.916 | 0.612 | 69% |
| Without satellite | 20 | 0.900 | 0.528 | 64% |
| Without roads and people | 21 | 0.899 | 0.575 | 66% |
| Without province | 23 | 0.916 | 0.607 | 68% |
| Only weather | 13 | 0.764 | 0.325 | 41% |
| Only terrain | 3 | 0.765 | 0.306 | 42% |
| Only satellite | 4 | 0.739 | 0.387 | 55% |
| Only roads and people | 3 | 0.860 | 0.455 | 59% |
| Only province | 1 | 0.759 | 0.296 | 38% |

Road distance and the two population counts do the most work for ROC-AUC. Satellite adds the most for PR-AUC. Weather adds little once the other groups are present (0.922 vs 0.917 without it). The groups overlap, so the drops do not add up, and differences under about 0.01 AUC may be noise (one train/test split). Roads and people probably measure how remote a fire is; that reading is not tested.

**Against the Canadian fire weather index (FWI).** FWI columns exist for 2012-2025 (95% of 2025 fires), so this test uses 3,433 fires from 2025. Baselines were trained on 2012-2024 fires that have FWI.

| Score | ROC-AUC | PR-AUC | Big fires in top 15% |
|---|---|---|---|
| Model v14.6 (v4) | 0.920 | 0.588 | 74% |
| 13 weather columns only (logistic) | 0.719 | 0.205 | 37% |
| FWI columns only (logistic) | 0.673 | 0.135 | 24% |
| Best single FWI column (BUI, 7-day mean) | 0.617 | 0.110 | 17% |
| Random guessing | 0.50 | 0.087 | 15% |

FWI is built to describe fire danger, not how big a fire becomes after it starts, so a weak result here is not a fault of FWI. The baselines are simple logistic models and were trained on fewer fires than the main model. Scripts: `analysis/v14_6/ablation_feature_groups.py` and `analysis/v14_6/baseline_fwi_weather.py`.

## Satellite timing audit (2026-10-01)
Question: the satellite window includes the report day, and report dates have no clock time. Do report-day detections carry the result? The four satellite features were rebuilt from the raw detections in the database for the 2025+ test fires, with and without the report day.

| 2025+ test fires | ROC-AUC | PR-AUC | 0.770 rule: caught |
|---|---|---|---|
| Stored features (reported scores) | 0.923 | 0.621 | 72% |
| Rebuilt, report day included | 0.914 | 0.582 | 66% |
| Rebuilt, report day removed | 0.899 | 0.524 | 58% |
| All 4 satellite features set to 0 | 0.900 | 0.535 | 57% |

- **Finding:** the whole satellite gain comes from detections on the report day. Detections strictly before the report day add nothing (0.899 vs 0.900 with no satellite).
- 60% of fires with a detection have it only on the report day. Those fires are big more often (42%) than fires with an earlier detection (31%).
- **What it means:** if the model is scored before that day's satellite pass is known, expect about ROC-AUC 0.90 and PR-AUC 0.53 (the "Without satellite" ablation gave 0.900 and 0.528). The 0.923 / 0.621 headline holds only when same-day detections are really available at scoring time. For 2026, the report date is the first day a fire appeared in the CWFIS feed, which is itself built from satellite hotspots, so same-day detections there may be partly circular.
- **Limit:** the database holds only about half of the stored detections (rebuilt count above 0 for 644 MODIS fires vs 1,306 stored), so the rebuilt rows are a lower bound.
- Script: `analysis/v14_6/satellite_timing_audit.py`.

### Model without satellite features (known before the report day)
Same settings, 20 features (the four satellite columns removed). Trained to 2024, tested on 2025+ (v4). Script: `analysis/v14_6/no_satellite_model.py`.

| 2025+ | Full model (24) | No satellite (20) |
|---|---|---|
| ROC-AUC | 0.923 (0.916-0.930) | 0.901 (0.893-0.909) |
| PR-AUC | 0.621 (0.593-0.650) | 0.531 (0.506-0.562) |
| 2025 only: ROC-AUC / PR-AUC | 0.915 / 0.582 | 0.887 / 0.457 |
| 2026 only: ROC-AUC / PR-AUC | 0.929 / 0.656 | 0.912 / 0.598 |
| Alert rule from validation | raw >= 0.770: flags 16%, catches 72%, precision 52% | raw >= 0.688: flags 18%, catches 71%, precision 46% |
| Top 15% by score | catches 69%, precision 53% | catches 65%, precision 50% |
| Top 5 / 10 / 20 / 25% capture | 32 / 54 / 80 / 87% | 28 / 49 / 75 / 83% |

The no-satellite model needs its own threshold (0.688, not 0.770), and the existing calibrator was not fitted for it. Files (on Drive only): `final_model_v14.6_nosat_allyears.pkl`, `final_model_v14.6_nosat_info.json`. The 2025-only PR-AUC (0.457) is clearly lower than 2026 (0.598), so same-day satellite data helps most in 2025.

## Test: would knowing the next 3 days of weather help? (2026-10-03)
Upper-bound test with ACTUAL weather for the report day and the next 2 days (a perfect forecast, not a real one). No-satellite model, same fires with and without 6 new columns. Sample: 6,000 training fires (random, 2010-2024) and 4,000 test fires (2025+, 11.6% big). Script: `analysis/v14_6/forecast_upper_bound.py`.

| | ROC-AUC | PR-AUC | Top 15% catches |
|---|---|---|---|
| Without next-3-day weather | 0.868 | 0.464 | 58% |
| With next-3-day actual weather | 0.882 | 0.466 | 60% |
| Gain (95% range) | +0.013 (+0.007 to +0.020) | +0.001 (-0.020 to +0.023) | |

Even with a perfect forecast, PR-AUC did not move. ROC-AUC rose a little. A real forecast would gain less, so forecast weather was **not built**. Scores in this test are lower than the headline numbers because the model trained on 6,000 fires, not 131,000; compare only the two rows.

**Second run, whole years, weather from Earth Engine (2026-10-03).** Train on every 2024 fire (5,761), test on every 2025 fire (6,054, 10.0% big). Weather: ERA5-Land daily from Earth Engine, 5 new columns (highest temperature, strongest wind, average wind, total rain, lowest humidity; no gust). Script: `analysis/v14_6/forecast_upper_bound_ee.py`.

| | ROC-AUC | PR-AUC | Top 15% catches |
|---|---|---|---|
| Without next-3-day weather | 0.865 | 0.434 | 61% |
| With next-3-day actual weather | 0.876 | 0.462 | 62% |
| Gain (95% range) | +0.011 (+0.005 to +0.017) | +0.027 (+0.008 to +0.047) | |

Training on 2022-2024 (17,790 fires) and testing on 2025 gave the same picture: ROC-AUC 0.865 to 0.879 (+0.014, 0.009 to 0.020) and PR-AUC 0.429 to 0.455 (+0.026, 0.008 to 0.042). So the gain holds with more training data. This run shows a small gain, above zero, where the sample run showed none. Both are the best case (perfect forecast), so a real forecast should gain less, likely about +0.01 PR-AUC or less. Forecast weather is not built into the model.

## Raw NFDB column audit and cause test (2026-10-03)
Scripts: `analysis/v14_6/nfdb_column_audit.py`, `analysis/v14_6/cause_feature_test.py`.
- **Not usable:** `OUT_DATE`, `SIZE_HA`, `CFS_NOTE2` (filled in after the fire); `ATTK_DATE` (93-100% blank); `RESPONSE` (the agency's own decision, 85% blank, and a fire left to "monitor" is big 41% of the time because it is monitored, so using it would be circular).
- **Prescribed burns** (`PRESCRIBED` = PB, `CAUSE2` = H-PB) are about 800 fires out of 448,618 and are big 27-30% of the time. They are not wildfires and should be excluded when the data is next rebuilt.
- **Cause** (natural / human / unknown) separates big fires strongly on its own (2025+: 21.7% big for natural, 2.4% for human, 4.6% for unknown), but adding it to the no-satellite model gave **no gain**: ROC-AUC 0.901 to 0.902, PR-AUC 0.531 to 0.530 (gain -0.001, 95% range -0.006 to +0.003). The model already gets this information from road distance, population, province and weather, so cause is left out and its timing does not matter.

## Satellite benefit by time of day (2026-10-03)
Report times are not in the data, so this shows the whole curve: the full model (trained with the whole report day) is scored on 2025+ fires, but only detections before local hour H on the report day are visible (local time approximated from longitude). Script: `analysis/v14_6/satellite_cutoff_sensitivity.py`. The rebuilt rows use only the detections stored in the database (about half of those behind the stored features), so they are a lower bound.

| Detections visible before (local) | ROC-AUC | PR-AUC | 0.770 rule catches |
|---|---|---|---|
| No satellite at all | 0.900 | 0.535 | 57% |
| 00:00 (none from the report day) | 0.900 | 0.530 | 59% |
| 09:00 | 0.901 | 0.537 | 59% |
| 12:00 | 0.903 | 0.544 | 60% |
| 15:00 | 0.914 | 0.582 | 66% |
| 18:00 | 0.914 | 0.583 | 66% |
| 21:00 | 0.918 | 0.590 | 67% |
| 24:00 (whole report day) | 0.919 | 0.591 | 67% |
| Stored features (as reported) | 0.923 | 0.621 | 72% |

- The earliest report-day detection is at a local solar hour of 13 (median); 90% are by hour 15. These are the afternoon satellite passes.
- **Before noon, satellite adds nothing. From mid-afternoon on, it adds about +0.05 PR-AUC** (more with the full detection set).
- **Operational rule:** use the no-satellite model for fires scored before about 15:00 local on the report day. Use the satellite model from about 15:00 on the report day, or re-score the next morning once the day's detections are in.

## Lightning test (2026-10-03)
CanCPLD cloud-to-ground flashes (0.1 degree, 3-hourly, 2004-2024). Five columns: flashes in the 3x3 cells around the fire over the 1, 3 and 7 whole UTC days before the report day, and in the 21x21 cells (about 2 degrees) over 3 and 7 days. Nothing from the report day is used. No-satellite model, train 2004-2021, test 2022-2024 (18,010 fires). The data ends in 2024, so 2025+ could not be tested. Script: `analysis/v14_6/lightning_feature_test.py`.

| | ROC-AUC | PR-AUC | Gain (95% range) |
|---|---|---|---|
| Without lightning | 0.894 | 0.527 | |
| With lightning | 0.895 | 0.532 | ROC-AUC +0.000 (-0.001 to +0.001), PR-AUC +0.005 (0.000 to +0.010) |
| Natural-cause fires only, without | 0.839 | 0.537 | |
| Natural-cause fires only, with | 0.840 | 0.543 | PR-AUC +0.006 (-0.000 to +0.011) |

Fires with lightning nearby in the previous week are big about twice as often (0 flashes: 8.1%, 1-50 flashes: 16-17%), but the model already gets most of that from its other columns. The gain is too small to justify a live lightning feed, so lightning is left out.

## Ensemble test (2026-10-03)
No-satellite model (20 features, v4). Five models (LightGBM, bagged LightGBM, XGBoost, CatBoost, scikit-learn HistGradientBoosting), combined by equal-weight rank average, nothing tuned on the test years. Script: `analysis/v14_6/ensemble_test.py`.

| PR-AUC | Validation 2022-24 | Forward test 2025+ |
|---|---|---|
| LightGBM (current) | 0.527 | 0.531 |
| Best single other model | 0.531 (bagged LightGBM) | 0.542 (XGBoost) |
| Ensemble of all 5 | 0.530 | 0.539 |
| Ensemble of LightGBM + XGBoost + CatBoost | 0.528 | 0.542 |
| Gain of the 3-model ensemble (95% range) | +0.002 (-0.003 to +0.007) | +0.011 (+0.004 to +0.017) |
| Gain of the 5-model ensemble (95% range) | +0.003 (-0.002 to +0.008) | +0.007 (0.000 to +0.014) |

The gain is about +0.01 on the forward test and about zero on the validation years, so it is not consistent. Both ensembles were written into the script before it was run, but the 3-model one is the best of the two on 2025+, and 2025+ has already been used for other decisions, so treat +0.011 as exploratory, not a clean result. Re-test the chosen ensemble on 2027 before using it as the headline. The ensemble costs three to five times the compute and complexity at scoring time, so the single LightGBM stays as the main model.

## Final QA audit (2026-09-29) - 14 of 14 checks passed
- The saved model file reproduces the reported scores exactly. Its feature names, order and types match the info file.
- No feature is a size, end-date, year or match column.
- **With and without satellite (2025+):** with a detection (1,437 fires, 38% big): ROC-AUC 0.852, PR-AUC 0.743. Without one (7,127 fires, 6% big): ROC-AUC 0.912, PR-AUC 0.430. The model still ranks well when no satellite saw the fire.
- **By cause (2025+):** natural fires (N): ROC-AUC 0.879, PR-AUC 0.663 (881 of the 974 big fires, 22% big). Human fires (H): ROC-AUC 0.878, PR-AUC 0.209 (only 62 big, 1.6% big). Unknown (U): 0.933 and 0.536 (31 big, 4% big). PR-AUC depends on how common big fires are, so these three PR-AUC values cannot be compared directly.
- **Sensor eras (leave-one-year-out, not forward-looking):** average ROC-AUC is 0.913 for 2004-2011 (MODIS only), 0.916 for 2012-2021 and 0.912 for 2022-2024. Nothing odd when VIIRS appears. The recovered years 2006 (0.881) and 2007 (0.937) look like any other year.
- **Direction checks (not a strict test):** hotter, drier, windier or less rain all raised the average score, and removing satellite detections lowered it. This shows the average moves the right way, not that every fire does. The effects are small (only 49% of fires went up when hotter, 46% with less rain), so most of the ranking comes from other inputs.

### By province (2025+, v3, not rerun)
| Province | Fires | Big | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| AB | 1,250 | 19 | 0.932 | 0.281 |
| BC | 2,596 | 144 | 0.845 | 0.293 |
| MB | 289 | 120 | 0.814 | 0.743 |
| NL | 352 | 31 | 0.935 | 0.732 |
| NT | 424 | 218 | 0.797 | 0.771 |
| ON | 763 | 82 | 0.898 | 0.531 |
| QC | 1,108 | 152 | 0.932 | 0.590 |
| SK | 552 | 115 | 0.879 | 0.619 |
| YT | 230 | 61 | 0.901 | 0.723 |
| Parks Canada (province code 8) | 116 | 25 | 0.956 | 0.863 |

NB (4 big fires), NS (3) and PE (0) are too small to judge. No confidence ranges were computed per province, so treat these numbers as rough.

## Data audit (2026-09-29)
- **Values:** no impossible values in any feature (weather, NDVI, elevation, slope, population, satellite). VIIRS is exactly 0 (not empty) before 2012.
- **Recovered 2006-07 rows** look like the years around them (all medians within 0.15 of the spread).
- **No odd year-to-year jumps** in any checked feature.
- **Dates:** the 2026 report dates are fine ("2026-05-14 00:00:00"). Some tools fail to read them only because of the extra time part. The capital-letter `YEAR`, `MONTH`, `DAY` columns are empty for 2012+ (the lowercase `year` is complete). The model does not use them.
- **UNIQUE_ID** is empty for 2012+ (85,264 rows). Uniqueness could only be checked where it is filled (2004-2011): no duplicates there.
- **Province code 8** is Parks Canada fires (names mixed AB, BC, NT, PC).
- **Small issues, left as is:** 10 fires have a road distance over 1,000 km (2 are (0,0) coordinates, 2 are in the ocean, 6 have real Canadian coordinates but a failed road lookup). Set these to empty before scoring; LightGBM handles empty values; 19 fires have a MODIS count above 0 with strength 0 (whether that 0 is real or a failed reading was not settled); 0.85% of fires share the same date and place as another fire (may be real repeat reports).
- **Not yet checked:** raw satellite timestamps and spatial matching (need the raw MODIS/VIIRS files), and whether population is per year or one modern layer.

## Left out on purpose
- **FWI (fire-weather index):** tested twice, no benefit.
- **`YEAR_clean`:** a year column stops the model handling new years.
- **`n_modis_matches`, `has_modis_match`:** they count satellite detections over the whole fire, which leaks the final size. They are also empty for 2012+ fires.
- **`sensor_available`, `sat_zero`:** no real gain (tested with vs without on the final 24 features). `sat_zero` is unreliable (all 1 in 2025, all 0 in 2026).
- **Water distance, settlement distance, road km, nearby-fire counts, month/day, Alberta flag:** best gain +0.003, below the 0.01 bar.
- Also rejected earlier: multi-VIIRS variants, fuel type, PCA, extra tuning.

## Known limits
- **Satellite features depend on same-day detections.** Without the report day, the satellite gain disappears (see Satellite timing audit). Quote 0.900 / 0.528 as the score for "known before the report day" and 0.923 / 0.621 as "with same-day satellite data".
- **Inside one province, the ranking is weaker than the national score.** The national ROC-AUC of 0.923 is helped by the model knowing which provinces have more big fires. Within a single large province, expect ROC-AUC of about 0.80 to 0.93 (v3 numbers: BC 0.845, MB 0.814, NT 0.797, SK 0.879, ON 0.898). The v4 rerun gave BC 0.835, NT 0.821, SK 0.841.
- **Human-caused fires are harder.** Their ranking works (ROC-AUC 0.878), but few are big, so precision is low (PR-AUC 0.209).
- **Big fires are rare in AB and BC,** so their PR-AUC is low (about 0.29) even though the ranking is good.
- VIIRS starts in 2012, so VIIRS values are 0 for older fires. The model has no flag for this and works fine without one.
- Only about 17% of 2025+ fires have any satellite detection in the window (4-7% before 2012 with MODIS only).
- In v3, columns the model does not use (FWI columns, `sat_zero`, water/settlement distance, road km, nearby-fire counts) are blank for the recovered 2006-07 fires.
- **Road and population features may partly capture how fires are managed.** In a 2004-2011 test (fires with a protection-zone name; train 2004-09, test 2010-11, 203 big fires), removing roads and population lowered ROC-AUC from 0.900 to 0.826. Adding the protection zone brought it back to 0.877, and the full model with the zone reached 0.906. So the zone explains a substantial part of what roads and population carry (about two-thirds of the gap), but they still add some signal beyond it. Zone alone reached 0.794 and roads/population alone 0.820. The sample is small, so the 0.03 left over could be noise. Protection-zone data exists only for some provinces in 2004-2011 and is empty from 2012 on, so this cannot be tested on the 2025-26 years. These are diagnostic results, not proof of cause: roads and population may also reflect accessibility, human activity, reporting delay, suppression or other geographic differences. They are kept as predictive proxies because they are known at report time.
- **Roads and population inside a province (v3 data, 2025-26, province from `province_encoded`; not rerun).** A v4 rerun of the simpler province test printed only BC (+0.102), NT (+0.032) and SK (+0.013); why other provinces were missing from that printout has not been checked, so the ranges below stay as the v3 reference. Gain in ROC-AUC from adding roads and population, with 95% ranges from 200 resamples: BC +0.068 (0.031 to 0.102), SK +0.051 (0.022 to 0.080), YT +0.051 (0.015 to 0.096), QC +0.028 (0.010 to 0.046), ON +0.019 (0.001 to 0.039). The gain is not clearly above zero in AB (+0.037, -0.012 to 0.105), MB, NL, NT and Parks Canada. So roads and population add signal beyond province in several provinces, most in BC, SK and YT, and little or none in others. Samples are small in some provinces (AB has only 19 big fires).
- **Gaps in the unified file: repaired in v4.** v3 was missing 2,584 fires in 2025 (for example Manitoba and Ontario had 0). The cause was an inner join on weather; see Data repair. Still missing: Yukon 2022 (289 fires, no report date). The `resolved_province` column is empty for all 2026 fires, so use `province_encoded` for province work.
- This is a model that ranks reported fires by how likely they are to end up big. It is not a physical fire-growth model.

## Load and use
```python
import joblib, json
from modeling.calibration import load_calibrator, apply_calibrator

folder = "models/"   # or your Drive folder
model = joblib.load(folder + "final_model_v14.6_clean_allyears.pkl")
info  = json.load(open(folder + "final_model_v14.6_clean_info.json"))
scores = model.predict_proba(new_fires[info["features"]])[:, 1]
alert  = scores >= info["threshold_recall"]          # 0.770 on the raw score

# Optional: an estimated chance for display
cal = load_calibrator(folder + "final_model_v14.6_calibrator.json")
chance = apply_calibrator(cal, scores)
```
Live scoring of a new fire report: `live_scoring/score_live_fire.py` (tests in `tests/`).
