# Model card - Canada wildfire "big fire" model, v14.6

Date: 2026-09-29 (final QA audit, data audit and calibration added)

## What it does
Predicts which Canadian wildfires will become big fires (more than 100 ha), using satellite, weather, terrain, road and population data for each fire. It uses only information from before or on the report day.

## Files (in the wildfire_project folder on Drive)
| File | Use |
|---|---|
| `final_model_v14.6_clean_allyears.pkl` | **Final model for live use** (trained on all years) |
| `final_model_v14.6_clean.pkl` | Same model trained through 2024. Gave the test scores below. Use for reports. |
| `final_model_v14.6_clean_info.json` | Feature list, threshold, top-15% setting |
| `final_model_v14.6_calibrator.json` | Turns the raw score into a real chance (see Calibration). Plain numbers, no pickle. |
| `unified_dataset_2004_2026_FINAL_v3.csv` | Training data |

## Data
- 140,406 fires, 2004-2026. The 2006 and 2007 fires (16,613) were missing from v2 and are now recovered.
- 149 fires with impossible coordinates (mostly 0,0) are left out of training and testing.
- Big fires are about 7.7% of all fires, and 11.4% in 2025+.
- The 2026 fires were built separately from the live CWFIS feed (per project notes): the report date is the first day a fire appeared in the feed, so it is approximate, and small fires still burning were left out. That makes the 2026 big-fire rate (13.5%) a little high.

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

## Scores (with 95% ranges from 500 bootstrap resamples)
| Data | Fires (big) | ROC-AUC | PR-AUC |
|---|---|---|---|
| **Test 2025+** | 8,564 (974) | **0.929** (0.921-0.936) | **0.634** (0.600-0.668) |
| 2025 only | 3,603 (302) | 0.923 (0.907-0.937) | 0.587 (0.530-0.651) |
| 2026 only | 4,961 (672) | 0.929 (0.921-0.937) | 0.656 (0.617-0.693) |
| Validation 2022-2024 | | 0.909 | 0.599 |

The 2025 and 2026 PR-AUC ranges overlap, so the gap between those years may be noise. Random guessing would give a PR-AUC of about 0.114.

| Model (same 2025+ fires) | ROC-AUC | PR-AUC |
|---|---|---|
| v14.6 | 0.929 | 0.634 |
| v14.5 | 0.927 | 0.625 |
| v14.1 | 0.922 | 0.607 |
| v14.4 (has leaky columns, do not use) | 0.916 | 0.572 |

## Alert rules (on 2025+)
| Rule | Big fires caught | Flagged fires that are truly big | Share of fires flagged |
|---|---|---|---|
| Score at or above 0.770 | 73% | 53% | 16% |
| Top 15% by score | 71% | 54% | 15% |

**By year:**
| | 0.770 rule: flagged / caught / precision | Top 15% rule: caught / precision |
|---|---|---|
| 2025 | 11% / 65% / 49% | 76% / 42% |
| 2026 | 19% / 76% / 55% | 67% / 61% |

With the fixed 0.770 rule, the share flagged follows how bad the season is (11% in a milder year, 19% in a bad one). That is expected, not a fault. The top-15% rule always flags 15%, but recall and precision then move by year.

**Top-risk fires (2025+):** the top 5% of fires by score contain 33% of the big fires, the top 10% contain 55%, the top 15% contain 71%, the top 20% contain 82%, and the top 25% contain 89%.

## Calibration (2026-09-29)
The calibrator (Platt, a simple curve) was fitted on 2022-2024 scores from a model trained only to 2021, then tested on 2025+. It does not change the ranking, so ROC-AUC and PR-AUC stay the same.

| 2025+ | Average score | Actual rate | Brier (lower is better) | Slope (1.0 = perfect) |
|---|---|---|---|---|
| Raw score | 0.269 | 0.114 | 0.114 | 0.87 |
| Calibrated | 0.118 | 0.114 | **0.061** | **1.00** |
| Guessing the overall rate | | | 0.101 | |

- 2025: calibrated average 0.088 vs actual 0.084. 2026: 0.140 vs 0.135. The calibrator held across both very different seasons.
- Reliability bins: the biggest gap between predicted and actual in any group was 2.3 points; the average gap was 0.7 points.
- **What the numbers mean:** raw score 0.770 is about a 29% chance; the top 5% of fires are about 64% or more; the median fire is about 2%.
- The saved 2024-trained model with the same calibrator also worked (average 0.120 vs actual 0.114).
- **Limits:** the calibrator was fitted on 2022-24 and can drift in a very different season. It was fitted for the 2024-trained model's kind of scores, so for the all-years model it is a close approximation, not a tested match. Refit it when 2027 fires are in. Small provinces and rare groups (for example human-caused fires) were not checked one by one.
- Use the calibrated chance only for showing a risk number. The alert rules (0.770 raw score, or top 15%) stay as they are.

## Final QA audit (2026-09-29) - 14 of 14 checks passed
- The saved model file reproduces the reported scores exactly. Its feature names, order and types match the info file.
- No feature is a size, end-date, year or match column.
- **With and without satellite (2025+):** with a detection (1,437 fires, 38% big): ROC-AUC 0.852, PR-AUC 0.743. Without one (7,127 fires, 6% big): ROC-AUC 0.912, PR-AUC 0.430. The model still ranks well when no satellite saw the fire.
- **By cause (2025+):** natural fires (N): ROC-AUC 0.879, PR-AUC 0.663 (881 of the 974 big fires, 22% big). Human fires (H): ROC-AUC 0.878, PR-AUC 0.209 (only 62 big, 1.6% big). Unknown (U): 0.933 and 0.536 (31 big, 4% big). PR-AUC depends on how common big fires are, so these three PR-AUC values cannot be compared directly.
- **Sensor eras (leave-one-year-out, not forward-looking):** average ROC-AUC is 0.913 for 2004-2011 (MODIS only), 0.916 for 2012-2021 and 0.912 for 2022-2024. Nothing odd when VIIRS appears. The recovered years 2006 (0.881) and 2007 (0.937) look like any other year.
- **Direction checks (not a strict test):** hotter, drier, windier or less rain all raised the average score, and removing satellite detections lowered it. This shows the average moves the right way, not that every fire does. The effects are small (only 49% of fires went up when hotter, 46% with less rain), so most of the ranking comes from other inputs.

### By province (2025+)
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
- **Inside one province, the ranking is weaker than the national score.** The national ROC-AUC of 0.929 is helped by the model knowing which provinces have more big fires. Within a single large province, expect ROC-AUC of about 0.80 to 0.93 (BC 0.845, MB 0.814, NT 0.797, SK 0.879, ON 0.898).
- **Human-caused fires are harder.** Their ranking works (ROC-AUC 0.878), but few are big, so precision is low (PR-AUC 0.209).
- **Big fires are rare in AB and BC,** so their PR-AUC is low (about 0.29) even though the ranking is good.
- VIIRS starts in 2012, so VIIRS values are 0 for older fires. The model has no flag for this and works fine without one.
- Only about 17% of 2025+ fires have any satellite detection in the window (4-7% before 2012 with MODIS only).
- In v3, columns the model does not use (FWI columns, `sat_zero`, water/settlement distance, road km, nearby-fire counts) are blank for the recovered 2006-07 fires.

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
