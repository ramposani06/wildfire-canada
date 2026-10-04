# Model card - Canada wildfire "big fire" model, v14.6

Date: 2026-10-01 (rerun on repaired data v4: scores, alert rules, calibration, ablation, FWI baseline; sections still on v3 are marked)

## Main model (decision 2026-10-01)
**The main model is the no-satellite model (20 features).** It uses only information known before the report day: ROC-AUC 0.901 (0.893-0.909), PR-AUC 0.531 (0.506-0.562) on 2025+. Alert threshold on its raw score: **0.688** (not 0.770). The full 24-feature model (0.923 / 0.621) is kept as an **optional satellite-enhanced model**, to be used only when satellite detections are timestamped and known to be earlier than the scoring time. Reason: the whole satellite gain comes from report-day detections whose timing relative to the report is unknown (see Satellite timing audit). The sections below that describe "the model" and the 0.770 rule refer to the full 24-feature model unless they say otherwise.

## Update: v14.7 (2026-10-03) - main model is now 22 features
**Main model = v14.7: the 20 no-satellite features plus latitude and longitude.** Single LightGBM, no ensemble. Train to 2024, one test on 2025+ (11,145 fires):
| | v14.6 no-satellite | v14.7 |
|---|---|---|
| ROC-AUC | 0.901 (0.893-0.909) | **0.904 (0.897-0.912)** |
| PR-AUC | 0.531 (0.506-0.562) | **0.541 (0.513-0.571)** |
| 2025 / 2026 PR-AUC | 0.457 / 0.598 | 0.471 / 0.599 |
| Alert threshold (65% recall on 2022-24) | 0.688 | **0.695** |
| Flags / catches / precision | 18% / 71% / 46% | 17% / 70% / 47% |
- Top 5/10/15/20/25% capture: 29 / 50 / 66 / 75 / 83%.
- Own Platt calibrator (coef 0.8972, intercept -1.7637, fitted on 2022-24 scores of the model trained to 2021). Brier 0.120 to 0.069. Average chance 12.3% vs actual 11.5%. The 0.695 threshold is about a 26% chance.
- Script: `analysis/v14_6/build_v14_7.py`. Files on Drive: `final_model_v14.7_allyears.pkl`, `final_model_v14.7_info.json`.
- The monitoring dashboard still shows the 20-feature version until its export is rerun with `WF_VARIANT=loc`. Sections below that say "no-satellite model" or "0.688" describe v14.6 unless they say otherwise.

### Exact location and other size cutoffs (`analysis/v14_6/location_and_cutoff_test.py`)
- Adding latitude and longitude to the 20 features (train to 2021): PR-AUC +0.012 on 2022-24 (range +0.004 to +0.021) and +0.014 on 2025+ (+0.004 to +0.026). Small but positive in both periods, so it was adopted.
- Other cutoffs, same 20 features, 2025+: PR-AUC is higher for smaller cutoffs only because more fires qualify. Lift over a random guess is steady (>10 ha 3.8x, >100 ha 4.6x, >500 ha 5.1x, >1000 ha 4.9x) and ROC-AUC stays 0.89-0.90. The 100 ha line is not what limits the score.

### Aspect and wind direction (rejected)
Eight columns: aspect (sin, cos) from the 30 m Copernicus DEM, wind direction, steadiness and speed from ERA5-Land for the 3 whole days before the report day, "wind blowing uphill" and uphill wind x slope. Script: `analysis/v14_6/aspect_wind_test.py`.
- A first small test (train 2022-24, test 2025 only, 6,054 fires) looked good: PR-AUC +0.020 (+0.004 to +0.036). With lat/lon it fell to +0.010 (-0.003 to +0.024).
- The full test (same protocol as the main model) shows nothing. Train 2004-2021, test 2022-24: -0.001 (range -0.006 to +0.005). Train 2004-2024, test 2025-26: +0.002 (-0.005 to +0.008). With lat/lon in the model: -0.006 (-0.011 to 0.000) and -0.004 (-0.011 to +0.002). Aspect alone: +0.000 and +0.001.
- Big-fire rate by uphill or downhill wind on slopes over 5 degrees is flat (about 6-9%, no trend).
- Lesson: a gain seen in one small sample (+0.020) did not survive the full test. Not added.

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

## Feature importance, all columns (2026-10-03)
Scripts: `analysis/v14_6/feature_importance_all.py`, `analysis/v14_6/feature_importance_everything.py`. Model trained to 2024, scored on 2025+ (11,145 fires). The data file holds 51 usable columns (22 in v14.7 plus 29 extra); 26 more were skipped (IDs, text, dates, and fields filled in after the fire).

**The 22 v14.7 features, three views.** "Shuffle" = PR-AUC lost when one column is scrambled on the test fires. "Drop" = PR-AUC lost when the model is retrained without it. Many columns carry the same information, so the drop numbers are small even where the shuffle numbers are large.
| Feature | Group | Shuffle | Drop |
|---|---|---|---|
| LATITUDE | location | 0.089 | 0.012 |
| dist_to_road_m | roads & people | 0.055 | 0.002 |
| province_encoded | land | 0.036 | 0.005 |
| pop_within_10km | roads & people | 0.032 | -0.001 |
| pop_within_25km | roads & people | 0.030 | -0.001 |
| NDVI | land | 0.022 | 0.013 |
| LONGITUDE | location | 0.019 | 0.001 |
| elevation | land | 0.004 | 0.003 |
| Each of the 13 weather columns | weather | 0.002 or less | within +-0.004 |

**Whole groups removed at once (retrain):** without weather PR-AUC 0.525 (-0.015), without roads & people 0.529 (-0.012), without location 0.531 (-0.010), without land 0.534 (-0.006). Weather columns look weak one by one but are the group that costs the most to lose, because they cover for each other.

**Satellite for reference (26 features):** PR-AUC 0.630. `modis_max_frp_early7d` is the top column by shuffle (0.061), but the satellite timing problem still applies.

**Re-testing the 29 extra columns** (FWI/DC/BUI/ISI/FFMC, cause, fire type, protection zone, national park, prescribed flag, month, day, day of year, Alberta flag, road length, settlement and water distance, nearby-fire counts, resolved province). Each was added alone to v14.7 on two forward splits (train to 2021 -> 2022-24, and train to 2024 -> 2025+):
- **None helped on both splits.** Best: cause (+0.003 and +0.002) and day of year (+0.001 and +0.003), both under the +0.003 bar on at least one split.
- All 29 together: PR-AUC 0.530 vs 0.541 for v14.7 (-0.011, 95% range -0.023 to +0.003). More columns made it slightly worse.
- **Caveat:** FWI, DC, BUI, ISI and FFMC are 63% empty, and road length, settlement and water distance and nearby-fire counts are 44% empty (they do not exist for all years). Their negative results on 2025+ may partly come from that missing data rather than from the columns being useless. Earlier tests on complete subsets also found no gain.
- Conclusion: v14.7 stays at 22 features.

## Combinations of weak features and clustering (2026-10-03)
Script: `analysis/v14_6/combo_cluster_test.py`. Model v14.7 trained to 2021; patterns searched in 2022-24 and then checked on 2025+. "Weak" = the 13 weather columns, elevation and slope.
- **Rules:** a depth-3 tree on what the model gets wrong gave 7 groups. None held up: the model's error in the best group fell from +3.9 points (2022-24) to +0.9 (2025+), in the next from +3.1 to +0.4, and one group flipped sign (-4.2 to +3.0). Noise.
- **Missed big fires:** 35% of big fires in 2022-24 and 31% in 2025+ score under the alert threshold. Compared with caught big fires they are further north, in provinces with lower codes, nearer roads and on steeper, higher ground. Clustering them gives four groups (near-road, cool and cloudy north, very dry north, steep high ground). The group sizes change between periods (for example 28% to 18% and 16% to 26%), so none is a stable pocket.
- **Cluster as a feature** (k-means on the weak features, k = 8, 20, 50, plus distance to the cluster centre): PR-AUC gain between -0.003 and +0.004 on both splits, every range crosses zero. Not added.
- Conclusion: no combination of the weak features adds reliable signal. The missed big fires look like unlucky growth rather than a hidden pattern.

## Missed big fires, profile (2026-10-03)
Script: `analysis/v14_6/missed_fires_profile.py`. Model v14.7 trained to 2021, threshold 0.695 (65% recall on 2022-24), scored on 2022-26 fires (29,155; 3,369 big). Caught 2,238, **missed 1,131 (34%)**, false alarms 2,362. Final size is used only to describe fires after the fact.
- **Close calls:** 33% of misses scored 0.55-0.70 (just under the line); 19% scored under 0.20.
- **Size:** catch rate is 53% for 100-200 ha fires, 59% for 200-500, and about 70% above 500 ha.
- **Where:** BC (77% of its big fires missed) and AB (67%) make half of all misses. NT 5%, YT 11%, SK 18%, QC 21% missed.
- **Road distance:** missed 87% under 1 km, 75% at 1-5 km, 47% at 5-20 km, 9% over 20 km.
- **Cause:** human-caused big fires missed 80%, natural 29%.
- **Month:** April 81% missed, May 62%, June 21%, July 32%, Aug 39%.
- **Typical values:** missed big fires have a median road distance of 5 km and 243 people within 25 km. Caught big fires have 40 km and 0 people. Weather is almost the same for all four groups.
- **Reading:** the model mostly learned that far from roads and people means big. Big fires near roads and people (mostly BC and AB, human-caused, spring) look like small fires to it, and weather alone does not separate them. The weak spot is exactly where a big fire matters most for people and property.

## Accessible-fire segment test (2026-10-03)
Script: `analysis/v14_6/accessible_segment_test.py`. Segment A = within 5 km of a road (70% of fires, 3.6% big, against 30% big elsewhere). Segment B = 1,000+ people within 25 km (55% of fires, 2.7% big). Two forward splits.
- **The national model ranks well inside the segment.** Segment A: ROC-AUC 0.869 (2022-24) and 0.878 (2025+); PR-AUC 0.24-0.25 on a 3.5% base rate, a lift of 7.0x (the national lift is 4.6x). Segment B: ROC-AUC 0.852 / 0.862, lift 5.8x / 7.2x.
- **A specialist model trained only on segment fires is not better:** PR-AUC gain -0.032 (-0.051 to -0.011) and -0.012 (-0.030 to +0.006) in segment A. Counting segment fires 3x also gave nothing (-0.011 and -0.005; segment B +0.006 and +0.008, ranges cross zero).
- **The problem is the alert line, not the ranking.** With one national line (0.695), the model catches 15% of big fires inside segment A and 4% inside segment B, although it ranks them well.
- **A separate line inside the segment** (65% recall on 2022-24 inside it; 0.51 for A, 0.48 for B), checked on 2025+: inside segment A it catches 43% instead of 15% (precision 27% instead of 42%); across all fires it catches 69% instead of 62% of big fires and flags 17% of fires instead of 14%. Segment B: inside it 40% instead of 4%; overall 67% instead of 62%; flags 16% instead of 14%. Tested on one forward period only.
- **Correction (see per-province test below):** this comparison did not give the national line the same alert budget, and its lines were fitted on scores the model had already trained on. The fair test found that separate lines do not catch more big fires than one national line flagging the same share of fires. A separate line near roads moves alerts to where coverage is thin; it does not add catches overall.

## Last three checks: interactions, size bands, subgroups (2026-10-03)
Script: `analysis/v14_6/last_three_checks.py`.
- **Interaction columns** (road or people x dryness, temperature, wind, slope, elevation, NDVI; NDVI x weather; slope x weather; 5 groups plus all together), added to v14.7 on two forward splits: PR-AUC gain between -0.004 and +0.005, every 95% range crosses zero. Trees already learn interactions; nothing added.
- **Score by final size (fires 2022-26, alert line 0.695).** Inside the near-road segment the median score is 0.04 for fires under 100 ha and 0.36-0.46 for big ones, but only 16-26% of big fires reach the line. Outside it (more than 5 km from a road) the median is 0.53 for small fires (33% above the line) and 0.86-0.89 for big ones (71-83% above the line). The score separates big from small in both places; the 100 ha line is where the classes overlap, and in accessible areas the whole big-fire group sits below the national line. The score does not rise further with size above 500 ha.
- **Ranking inside subgroups** (frozen model trained to 2021, 2022-26):
| Subgroup | Fires | Big | ROC-AUC | PR-AUC | Lift |
|---|---|---|---|---|---|
| All fires | 29,155 | 3,369 | 0.900 | 0.541 | 4.7x |
| Within 5 km of a road | 20,178 | 710 | 0.870 | 0.242 | 6.9x |
| BC | 7,103 | 532 | **0.797** | 0.244 | **3.3x** |
| BC + near road | 5,035 | 192 | **0.779** | 0.108 | **2.8x** |
| AB | 4,459 | 241 | 0.909 | 0.399 | 7.4x |
| AB + near road | 3,561 | 76 | 0.885 | 0.156 | 7.3x |
| Human-caused | 13,101 | 267 | 0.864 | 0.162 | 7.9x |
| April-May | 7,367 | 363 | 0.889 | 0.350 | 7.1x |
| Rest of Canada (not BC/AB) | 17,593 | 2,596 | 0.910 | 0.595 | 4.0x |
- **Reading:** in AB, human-caused fires, spring and near-road fires the ranking is good and the miss comes from the single alert line (see the accessible-fire test). **BC is a real ranking weakness**, worst near roads (ROC-AUC 0.78, lift 2.8x). That is where the available report-time data runs out.
- **v14.7 is frozen** as the main model. The remaining uncertainty is among fires with similar report-time conditions that grow differently.

## Other targets: smaller sizes and fire duration (2026-10-03)
Script: `analysis/v14_6/alternative_targets_test.py`. Same 22 features as v14.7, same two forward splits.
| Target | Test | Base rate | ROC-AUC | PR-AUC | Lift | Top 15% catches |
|---|---|---|---|---|---|---|
| Over 4 ha | 2022-24 | 22.3% | 0.859 | 0.672 | 3.0x | 49% |
| Over 4 ha | 2025+ | 21.9% | 0.879 | 0.712 | 3.3x | 52% |
| Over 10 ha | 2022-24 | 18.1% | 0.879 | 0.640 | 3.5x | 54% |
| Over 10 ha | 2025+ | 18.0% | 0.894 | 0.688 | 3.8x | 58% |
| Over 100 ha | 2022-24 | 11.6% | 0.898 | 0.539 | 4.7x | 64% |
| Over 100 ha | 2025+ | 11.5% | 0.904 | 0.541 | 4.7x | 66% |
- The model also ranks the smaller targets well (ROC-AUC 0.86-0.89), but its edge over a random guess is smaller (3.0-3.8x against 4.7x), because small fires are harder to tell apart. A target near 4-10 ha is closer to "got past the first attack" than 100 ha, but the data cannot say that for sure.
- **Duration could not be tested.** `OUT_DATE` is empty for every fire from 2012 to 2024 and 29% filled in 2025 (50-78% in 2004-2011). The 2025 sample (1,774 fires) is not representative and gave weak results (ROC-AUC 0.63-0.69), which should not be trusted. A duration or "escaped initial attack" label needs agency records.

## Training from 2012 only (VIIRS era) vs from 2004 (2026-10-03)
Script: `analysis/v14_6/viirs_era_test.py`. Same test fires in both setups. 2004-2021 has 113,683 training fires (6.3% big); 2012-2021 has 58,541 (7.5% big).
| Features | Test | Trained from 2004 (PR / ROC) | Trained from 2012 (PR / ROC) | 2012 vs 2004 |
|---|---|---|---|---|
| 22, no satellite | 2022-24 | 0.539 / 0.898 | 0.530 / 0.894 | -0.009 (-0.016 to -0.002) |
| 22, no satellite | 2025+ | 0.541 / 0.904 | 0.540 / 0.904 | -0.001 (-0.012 to +0.009) |
| 22 + VIIRS | 2022-24 | 0.592 / 0.910 | 0.587 / 0.906 | -0.005 |
| 22 + VIIRS | 2025+ | 0.601 / 0.918 | 0.602 / 0.917 | +0.001 |
| 22 + VIIRS + MODIS | 2022-24 | 0.608 / 0.913 | 0.592 / 0.908 | -0.016 |
| 22 + VIIRS + MODIS | 2025+ | 0.630 / 0.925 | 0.622 / 0.923 | -0.008 |
- Training from 2012 never helps: it is the same or slightly worse. Keeping 2004-2011 stays.
- The satellite rows are reference only (report-day timing risk). They still show that VIIRS alone adds about +0.05 to +0.06 PR-AUC and MODIS another +0.03.

## Leave-one-out by year and province (2026-10-03)
Script: `analysis/v14_6/leave_one_out_year_province.py`. Main model (v14.7, 22 features).
- **By year (rolling: train on all earlier years, test one year).** Yearly PR-AUC ranges from 0.33 (2018, ROC-AUC 0.840) to 0.62 (2014); ROC-AUC is 0.84-0.94. PR-AUC follows how many big fires there were: 2023 (15.7% big) 0.59 and 2026 (13.5%) 0.59, against 2018 0.33 and 2020 (2.6% big) 0.40. Lift is 3.8-15x. Pooled 2013-2026: ROC-AUC 0.903, PR-AUC 0.496.
- **Dropping one year from the pooled result** moves PR-AUC by at most 0.012 (2018 out: +0.012; 2023 out: -0.012; 2026 out: -0.011). No single year drives the score; the big-fire years lift it and 2018 lowers it.
- **Dropping one year from training** (train 2004-2024 without it, test 2025+): PR-AUC changes by 0.005 or less for every year (full training 0.5408). No single training year matters.
- **Province (2025+, provinces known):** BC ROC-AUC 0.82, PR-AUC 0.27; NT ROC-AUC 0.73 (base rate 38%, lift 1.4x); YT 0.77; SK 0.83; AB 0.86. NT, YT and SK have high PR-AUC because 25-38% of their fires are big, not because they rank well. Removing a province's fires from training changes its own PR-AUC by up to 0.07, but only where it has fewer than 15 big fires (noise).
- **Province x year (frozen model trained to 2021):** BC is the lowest every year (PR-AUC 0.18-0.30). NT 0.52-0.73, SK 0.54-0.70, YT 0.59-0.85, AB 0.36-0.51.
- **Data gap found:** `resolved_province` is empty for 7,542 of the 11,145 fires in 2025+ (all of 2026 and about 2,580 repaired 2025 fires). Province tables in this card that use it (missed-fire profile, subgroup ranking) therefore cover known-province fires only, mostly 2022-2024 plus part of 2025. The model itself uses `province_encoded`, which is filled for all fires. Fixed below (province filled and tables redone).

## Province names filled, province tables redone (2026-10-03)
Script: `analysis/v14_6/fill_province_and_redo.py`. `resolved_province` was empty for 62,838 of 142,838 fires: all of 2004-2011, about 2,580 repaired 2025 fires and all of 2026. The model was not affected (it uses `province_encoded`, filled for every fire).
- **Fill:** `province_encoded` maps to one province in 12 of its 13 codes (code 0 and code 8 both mean AB; one code mixes provinces and was filled by location). Nearest-fire fill by latitude and longitude is 99.3% accurate on known fires (5-fold). Where both methods can be used on an empty fire, they agree 99.6% of the time. The filled column is saved to Drive as `province_filled_v4.csv` (row number + province) for the next data rebuild.
- **By province, frozen model (train to 2021), fires 2022-26, alert line 0.695:**
| Province | Fires | Big | Base rate | ROC-AUC | PR-AUC | Lift | Catches | Precision | Fires alerted |
|---|---|---|---|---|---|---|---|---|---|
| BC | 8,588 | 611 | 7.1% | 0.797 | 0.232 | 3.3x | 21% | 28% | 5% |
| NT | 1,225 | 602 | 49.1% | 0.745 | 0.683 | 1.4x | 96% | 58% | 81% |
| SK | 2,379 | 437 | 18.4% | 0.877 | 0.595 | 3.2x | 81% | 46% | 32% |
| QC | 2,879 | 432 | 15.0% | 0.927 | 0.603 | 4.0x | 82% | 54% | 23% |
| MB | 1,599 | 419 | 26.2% | 0.805 | 0.567 | 2.2x | 73% | 49% | 39% |
| AB | 5,945 | 296 | 5.0% | 0.895 | 0.340 | 6.8x | 29% | 45% | 3% |
| YT | 631 | 246 | 39.0% | 0.824 | 0.709 | 1.8x | 89% | 59% | 59% |
| ON | 2,935 | 244 | 8.3% | 0.888 | 0.408 | 4.9x | 65% | 36% | 15% |
| NL | 693 | 62 | 8.9% | 0.912 | 0.654 | 7.3x | 74% | 61% | 11% |
- **Reading:** one national line behaves very differently by province. It alerts on 81% of NT fires and 59% of YT fires (half or more of them are big anyway) but on only 5% of BC and 3% of AB fires, so it catches 21% of BC and 29% of AB big fires. Ranking is best in QC (ROC-AUC 0.93), NL, AB, ON and SK; weakest in NT (0.70-0.75, lift 1.3-1.4x), MB (0.77-0.81) and BC (0.80). Per-province alert lines would fit better than one national line (see the accessible-fire test).
- **BC by year (PR-AUC, big fires):** 2022 0.18 (77), 2023 0.30 (252), 2024 0.24 (129), 2025 0.24 (82), 2026 0.15 (71). BC is the lowest every year. NT 0.52-0.74, SK 0.53-0.70, MB 0.30-0.65, ON 0.37-0.51, AB 0.20-0.51.
- **Final model (train to 2024) on 2025+:** BC ROC-AUC 0.814, PR-AUC 0.214; AB 0.848 / 0.191 (lift 5.9x); QC 0.941 / 0.619; SK 0.857 / 0.580; NT 0.703 / 0.660 (lift 1.3x); MB 0.766 / 0.621.
- The earlier missed-fire and subgroup tables used known-province fires only. With every province filled the picture is the same (BC and AB are where most big fires are missed).

## One national alert line vs a line per province (2026-10-03)
Script: `analysis/v14_6/province_threshold_test.py`. One frozen model (v14.7 trained to 2021). Each group's line is fitted on 2022-24 scores (65% recall inside the group; groups with under 40 big fires use the national line) and applied to 2025+ (11,145 fires, 1,283 big).
| Rule | Fires flagged | Big fires caught | Precision |
|---|---|---|---|
| One national line (0.695) | 16.3% | 68.7% | 48.6% |
| Line per province (8 groups) | 17.8% | 64.0% | 41.5% |
| National line moved to flag 17.8% (0.66) | 17.8% | **71.3%** | 46.2% |
| Line per province + near-road tier (11 groups) | 19.5% | 62.4% | 36.8% |
| National line moved to flag 19.5% (0.61) | 19.5% | **75.2%** | 44.3% |
- **Per-province lines are not more efficient.** For the same number of alerts, one national line catches more big fires (71.3% vs 64.0%; 75.2% vs 62.4%). Asking for equal recall everywhere spends alerts where precision is low.
- **What they do change is who is covered:** big fires caught in BC rise from 15% to 54% and in AB from 16% to 48%, while NT falls from 96% to 65% and QC from 90% to 88%. Alerted share in BC goes from 3% to 17% (precision 18%).
- **The per-province lines also do not hold their target:** 2025+ recall by province is 48-88% against a 65% target (AB 48%, BC 54%, QC 88%).
- **Reading:** this is a policy choice (equal coverage across provinces versus the most big fires caught per alert), not a model improvement. The national line alone is the better default.

## Over- and under-sampling (2026-10-03)
Script: `analysis/v14_6/resampling_test.py`. v14.7 features, two forward splits. The current model weights big fires up (`is_unbalance=True`).
| Setup | PR-AUC 2022-24 (gain) | PR-AUC 2025+ (gain) |
|---|---|---|
| Current (big fires weighted up) | 0.539 | 0.541 |
| No balancing | 0.533 (-0.005) | 0.544 (+0.003) |
| Undersample small fires 1:1, 5 draws | 0.536 (-0.003) | 0.547 (+0.006) |
| Undersample small fires 3:1, 5 draws | 0.539 (0.000) | 0.547 (+0.006) |
| Oversample big fires x3 | 0.538 (0.000) | 0.546 (+0.005) |
| SMOTE (synthetic big fires, 1:2) | 0.490 (-0.048) | 0.540 (-0.002) |
- Every 95% range for a non-SMOTE setup crosses zero, and none beats the current setup on both splits. SMOTE is clearly worse on 2022-24 (-0.048, range -0.058 to -0.038). Resampling changes the score scale (mean raw score 0.08-0.30), not the ranking, so the Platt calibrator is still needed. Current setup kept.

## What is missing for the fires the model cannot separate (2026-10-03)
Script: `analysis/v14_6/missing_information_test.py`. Fires 2022-25 (24,194; 2,697 big) scored by the frozen model (v14.7 trained to 2021). Look at the "unsure zone": fires with the same score where some became big and some did not. Each extra variable is tested alone (AUC inside the zone) and as a group boosted on top of the model score (leave-one-year-out). Most of these variables are not known at report time; the point is to find what kind of information is missing, not to ship it.
| Zone | Fires (big) | AUC of the model score inside the zone |
|---|---|---|
| Score 0.30-0.70, anywhere | 4,428 (673) | 0.605 |
| Score 0.20-0.70, within 5 km of a road | 3,280 (339) | 0.631 |
- **Single variables, inside the zone (AUC alone):** satellite detections are best (VIIRS max FRP 0.625-0.638, VIIRS and MODIS counts 0.62-0.63), better than the model's own score. Then road length within 5-10 km (0.39-0.44, more road = less likely big), distance to settlement (0.57-0.58), weather after the report (humidity min 0.42-0.43, rain 0.42-0.43, max temperature 0.55-0.56), and cause (human-caused 0.44-0.45).
- **Groups added on top of the score (gain in zone AUC):**
| Group | Anywhere | Near a road |
|---|---|---|
| All groups together | +0.093 (0.605 to 0.697) | +0.101 (0.631 to 0.732) |
| Satellite (7 days to end of report day) | +0.066 | +0.066 |
| Weather after the report (perfect forecast, 3 days) | +0.029 | +0.038 |
| Cause and time of year | +0.027 | +0.037 |
| Extra access and geography | -0.015 | -0.013 |
| Fire danger indices (FWI system) | -0.017 | -0.023 |
| Nearby fire activity | -0.022 | -0.029 |
| Lightning in the days before | -0.022 | -0.047 |
- **Reading:** what is missing is mostly (1) what the fire is doing in its first hours (satellite heat), (2) what the weather does after the report, and (3) cause and season. Nearby fires, lightning, danger indices and extra geography add nothing. Satellite detections include the report day, so the timing problem applies; weather after the report is only partly knowable from a forecast; the cause is not always known at report time (the whole-data test of cause found no gain).
- An earlier version of this test used a weaker baseline (0.536) and overstated every gain by about 0.07; it was replaced.

## Two-step scoring (report time, then re-score with satellite)

Step 1 = v14.7 at report time. Step 2 = same 22 features + 4 satellite columns, run after the day's satellite passes. Both frozen (trained to 2021), alert lines fitted on 2022-24 (step 1 = 0.695, step 2 = 0.770), scored once on 2025+ (11,145 fires, 1,283 big).

| Rule | Fires flagged | Big fires caught | Precision | False alarms |
|---|---|---|---|---|
| Step 1 only (report time) | 16.3% | 68.7% | 48.6% | 931 |
| Step 2 only (after satellite) | 15.7% | 71.4% | 52.3% | 837 |
| Step 1 OR step 2 | 18.4% | 76.7% | 47.9% | 1,069 |
| Step 1 AND step 2 | 13.6% | 63.4% | 53.8% | 699 |

- Ranking: step 1 ROC 0.904 / PR 0.548; step 2 ROC 0.924 / PR 0.622 (+0.07 PR).
- Step 1 missed 401 big fires; step 2 catches 102 of them (25%).
- Step 2 drops 300 step-1 alerts: 232 false alarms, 68 real big fires.
- Step 2 top-k capture: 5% 32%, 10% 55%, 15% 70%, 20% 81%, 25% 87%.

Reading: use step 1 as the alert at report time, then step 2 as a re-score to add late catches (OR rule: +8 points of recall for +2 points of fires flagged). Step 2 is only valid after the report day's satellite passes; it is not a report-time score. Single split, not repeated on 2022-24 or bootstrapped yet. Models saved on Drive: `final_model_v14.8_stage2_allyears.pkl`.

**Two-step check on two splits (bootstrap ranges).** Split A: train ≤2018, alert lines 2019-21, test 2022-24 (18,010 fires, 2,086 big). Split B: train ≤2021, lines 2022-24, test 2025+.

| | Split A | Split B |
|---|---|---|
| PR-AUC step 1 / step 2 | 0.540 / 0.610 | 0.548 / 0.622 |
| PR gain of step 2 | +0.070 (+0.057 to +0.082) | +0.074 (+0.056 to +0.093) |
| Recall step 1 / step 2 / OR | 69% / 76% / 78% | 69% / 71% / 77% |
| OR minus step 1: recall | +8.6 pts (+7.4 to +9.9) | +7.9 pts (+6.5 to +9.4) |
| OR: extra fires flagged | +3.1 pts | +2.2 pts |
| Precision step 1 / OR | 46% / 44% | 49% / 48% |
| Step 1 misses caught by step 2 | 28% (180 of 643) | 25% (102 of 401) |

Result: the gain holds on both splits and the ranges exclude zero. Step 2 adds about +0.07 PR-AUC and the OR rule adds about 8 points of recall for a 1-3 point precision cost. Still only valid after the report day's satellite passes.

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
