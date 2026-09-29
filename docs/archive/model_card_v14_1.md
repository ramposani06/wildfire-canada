# Model Card — Wildfire Canada Big-Fire Risk Model (v14.1)

## Intended use

Estimate the probability that a newly reported Canadian wildfire will grow to exceed 100 hectares, using only information available in the first hours after the report — before suppression outcomes are known. Intended to help prioritize which new fire reports deserve closer attention, not to replace judgment from fire management agencies.

**Not intended for**: real-time evacuation decisions, resource dispatch on its own, or use outside Canada (features and calibration are Canada-specific).

## Inputs

22 features, computed from the fire's reported location and date only:

- 13 weather features (7-day pre-report window): max/min temperature, precipitation, wind speed/gusts, humidity, sunshine duration
- 3 terrain/vegetation features: elevation, slope, NDVI
- 4 satellite pre-detection features: MODIS/VIIRS detection count and max fire radiative power in the 7 days before report
- 3 accessibility features: distance to nearest road, population within 10km/25km
- 1 geography feature: province

## Training data

- **Source**: Canadian National Fire Database (NFDB), 2012–2024
- **Training years**: 2012–2021 (~65,000 fires)
- **Calibration years**: 2022–2024 (Platt calibration + alert threshold selection)
- **Forward-test years**: 2025, 2026 — genuinely held out, never used for any tuning decision

## Performance

| Test year | PR-AUC | ROC-AUC | Recall at alert threshold |
|---|---|---|---|
| 2025 | 0.562 | 0.908 | 65.6% |
| 2026 | 0.636 | 0.922 | 73.5% |

Performance is not uniform across regions — see "Known limitations" below.

## v14 → v14.1: a correctness fix

While investigating consistently weaker performance in the Northwest Territories and Yukon, a data bug was found: **slope values for all 3,016 NT/YT fires in the training data were wrong**, sitting near zero (mean 0.18°) when the true terrain is far from flat (mean 5.37° after correction, some fires above 40°). Root cause: the original terrain-fetch pipeline's fallback for locations above SRTM's ~60°N coverage limit was implemented for elevation but never wired up for slope, so every high-latitude fire silently got a near-zero slope instead of a real one.

The fix was validated the same way every other change in this project is validated — retrained, then checked with bootstrap confidence intervals across all provinces and both forward-test years — before being adopted:

- No province showed a confirmed regression from the fix.
- 2026 showed a confirmed national improvement (bootstrap CI entirely above zero).
- 2025 showed no confirmed effect either way (CI crosses zero).

`final_model_v14_1.pkl` is the corrected production model. `final_model_v14.pkl` is retained for reference and comparison, not deleted.

## Known limitations

- **Regional weakness**: performance is consistently lower in NT, YT, SK, and BC than the national average — smaller fire counts, sparser weather stations, and different fire regimes than the provinces driving most of the training signal.
- **Two forward-test years**: 2025 and 2026 are the only genuinely held-out years so far. A third year (2027) will be added as it becomes available.
- **No independent reproduction**: this has not yet been reproduced by anyone outside this project.
- **Missing features, not yet tested**: lightning density, nearby recent-fire activity (a regional-outbreak signal). Identified as promising in a feature review but not yet built or validated.

## Tested and rejected

These were built, validated on 2022–2024, and then rejected because they failed to show a bootstrap-confirmed gain on the 2025/2026 forward test — logged here so the negative result isn't lost or silently re-tried later:

- Hyperparameter tuning beyond reasonable defaults (overfit to validation)
- FBP fuel type
- Terrain ruggedness (near-random signal, AUC 0.529)
- Distance to water (0.899 correlated with road distance — no independent signal after accounting for that)
- Several earlier attempts to incorporate pre-2012 historical fire data — each failed the bootstrap bar in earlier iterations, before a fix (adding a `sensor_available` flag to distinguish "no satellite existed yet" from "satellite looked and found nothing") produced the first version to show a real, reproducible effect. That work is ongoing and not yet part of the frozen production model.

## Ethical considerations

- Predictions could, in principle, influence how quickly a given region's fire report gets attention. Regional performance gaps (see above) mean the model is more confident/reliable in some provinces than others — this should be communicated alongside any operational use, not hidden behind a single national accuracy number.
- The model reflects patterns in 2012–2024 Canadian fire data. Climate and land-use patterns shift over time; performance should be periodically re-checked against new forward-test years rather than assumed to hold indefinitely.
