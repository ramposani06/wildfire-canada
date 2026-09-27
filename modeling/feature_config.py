"""
Single source of truth for feature lists and constants used across
training, evaluation, and live scoring. Import from here rather than
re-declaring feature lists in multiple scripts.
"""

# The 22 features the production model (v14.1) uses.
FEATURE_COLS_V14 = [
    # Weather — 7 days before official report date (13 features)
    "temperature_2m_max_mean",
    "temperature_2m_max_max",
    "temperature_2m_max_min",
    "temperature_2m_min_mean",
    "precipitation_sum_sum",
    "precipitation_sum_mean",
    "wind_speed_10m_max_mean",
    "wind_speed_10m_max_max",
    "wind_gusts_10m_max_mean",
    "relative_humidity_2m_mean_mean",
    "relative_humidity_2m_mean_min",
    "relative_humidity_2m_mean_max",
    "sunshine_duration_mean",
    # Terrain + vegetation (3 features)
    "elevation",
    "slope",
    "NDVI",
    # Satellite pre-detection, MODIS + VIIRS (4 features)
    "modis_count_early7d",
    "modis_max_frp_early7d",
    "viirs_count_early7d",
    "viirs_max_frp_early7d",
    # Accessibility (3 features)
    "dist_to_road_m",
    "pop_within_10km",
    "pop_within_25km",
    # Geography (1 feature)
    "province_encoded",
]

# Feature that flags whether VIIRS satellite coverage existed for this
# fire (1 for 2012+, 0 for pre-2012). Only used in the historical-data
# experiment (see notebooks/exploratory) — NOT part of FEATURE_COLS_V14.
SENSOR_AVAILABLE_FLAG = "sensor_available"

# Continuous features imputed with the training-set mean when missing.
MEAN_IMPUTE_COLS = ["NDVI", "elevation", "slope", "dist_to_road_m"]

# Features imputed with 0 when missing (absence is meaningful — e.g. no
# population data found nearby really does mean ~0 population).
ZERO_IMPUTE_COLS = ["pop_within_10km", "pop_within_25km"]

# A fire is labeled "big" if its final size exceeds this threshold, in hectares.
BIG_FIRE_THRESHOLD_HA = 100

# Temporal split — never changed after the model was frozen. Training
# only ever sees TRAIN_YEARS and CALIBRATION_YEARS; anything after that
# is a genuine forward test.
TRAIN_YEARS = (2012, 2021)
CALIBRATION_YEARS = (2022, 2024)

# Province code mapping (province_encoded), derived from the training
# data. Kept explicit here so it never silently drifts between scripts.
PROVINCE_MAP = {
    "AB": 0, "BC": 1, "MB": 2, "NB": 3, "NL": 4, "NS": 5,
    "NT": 6, "ON": 7, "PC": 8, "PE": 9, "QC": 10, "SK": 11, "YT": 12,
}
