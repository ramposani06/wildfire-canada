# What information is MISSING for the fires the model cannot separate?
# Take fires the model scores in the "unsure" range (look-alikes: same score, some became big, some did not).
# For each extra variable (not in the model), ask: inside that group, how well does it separate big from small?  (AUC 0.5 = not at all)
# Then add each group of variables to the score and see how much the in-group AUC rises (leave-one-year-out, 2022-25).
# Many of these variables are NOT usable at report time (weather after the report, report-day satellite). The point is to find WHAT is missing, not to ship them.
import os, json, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive")
CSV  = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
INFO = "final_model_v14.6_nosat_info.json"
LAT, LON, TARGET = "LATITUDE", "LONGITUDE", "is_big_fire"
P = dict(n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=31, is_unbalance=True, random_state=42, verbose=-1)
def find(name, must=True):
    for root, _, files in os.walk(FOLDER):
        if name in files: return os.path.join(root, name)
    if must: raise FileNotFoundError(name)
feats = json.load(open(find(INFO)))["features"] + [LAT, LON]
raw = pd.read_csv(find(CSV), low_memory=False); raw["_id"] = raw.index.astype(str)
df = raw[raw[LAT].between(41, 84) & raw[LON].between(-142, -52)].dropna(subset=[TARGET]).reset_index(drop=True)
y = df[TARGET].astype(int).values; yr = df["year"].values
m = LGBMClassifier(**P).fit(df.loc[yr <= 2021, feats], y[yr <= 2021]); df["score"] = m.predict_proba(df[feats])[:, 1]
# attach cached extras from earlier runs (matched on the original row number)
srcs = {"forecast_upper_bound_ee.csv": ["next3_temp_max", "next3_wind_max", "next3_wind_mean", "next3_precip_sum", "next3_rh_min"],
        "lightning_features_for_fires.csv": ["lt3_prev1", "lt3_prev3", "lt3_prev7", "lt21_prev3", "lt21_prev7"]}
for fn, cols in srcs.items():
    p = find(fn, False)
    if p: e = pd.read_csv(p).drop_duplicates("_id"); e["_id"] = e["_id"].astype(str); df = df.merge(e[["_id"] + cols], on="_id", how="left"); print(f"{fn}: found")
    else: print(f"{fn}: NOT found (those variables skipped)")
df["cause_natural"] = (df["CAUSE"].astype(str).str.strip().str[0] == "N").astype(float) if "CAUSE" in df.columns else np.nan
df["cause_human"] = (df["CAUSE"].astype(str).str.strip().str[0] == "H").astype(float) if "CAUSE" in df.columns else np.nan
df["rep_dayofyear"] = pd.to_datetime(df["REP_DATE"].astype(str).str[:10], errors="coerce").dt.dayofyear
GROUPS = {
 "weather AFTER the report (perfect forecast, next 3 days)": ["next3_temp_max", "next3_wind_max", "next3_wind_mean", "next3_precip_sum", "next3_rh_min"],
 "satellite detections (7 days to end of report day)": ["modis_count_early7d", "modis_max_frp_early7d", "viirs_count_early7d", "viirs_max_frp_early7d"],
 "lightning in the days before": ["lt3_prev1", "lt3_prev3", "lt3_prev7", "lt21_prev3", "lt21_prev7"],
 "nearby fire activity": ["fires_25km_7d", "fires_25km_14d", "fires_50km_7d", "fires_50km_14d"],
 "cause and time of year": ["cause_natural", "cause_human", "rep_dayofyear"],
 "fire danger indices (FWI system)": ["FWI_mean_7d", "FWI_max_7d", "DC_mean_7d", "BUI_mean_7d", "BUI_mean_30d", "ISI_max_7d", "FFMC_mean_7d"],
 "extra access / geography": ["dist_to_settlement_m", "dist_to_water_m", "road_km_within_5km", "road_km_within_10km"]}
for g in GROUPS: GROUPS[g] = [c for c in GROUPS[g] if c in df.columns]
d = df[(yr >= 2022) & (yr <= 2025)].copy(); d["big"] = d[TARGET].astype(int); d["acc"] = d["dist_to_road_m"] <= 5000
print(f"fires 2022-2025: {len(d):,} | big {int(d.big.sum()):,}")
ZONES = {"UNSURE zone, anywhere (score 0.30-0.70)": (d.score >= 0.30) & (d.score <= 0.70),
         "UNSURE zone, near a road (score 0.20-0.70, within 5 km of a road)": d.acc & (d.score >= 0.20) & (d.score <= 0.70)}
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 100)
for zn, zm in ZONES.items():
    z = d[zm].copy(); print(f"\n################ {zn}: {len(z):,} fires, {int(z.big.sum())} big ({z.big.mean():.0%}) ################")
    if z.big.sum() < 40: print("too few big fires"); continue
    sc_auc = roc_auc_score(z.big, z.score); print(f"The model's own score separates big from small inside this zone with AUC {sc_auc:.3f} (0.5 = nothing left)")
    rows = []
    for g, cols in GROUPS.items():
        for c in cols:
            v = z[c]; k = v.notna()
            if k.sum() < 150 or z.big[k].nunique() < 2: continue
            a = roc_auc_score(z.big[k], v[k]); sd = v[k].std() or 1
            rows.append((g.split(" (")[0][:34], c, int(k.sum()), a, (v[k & (z.big == 1)].mean() - v[k & (z.big == 0)].mean()) / sd))
    R = pd.DataFrame(rows, columns=["group", "variable", "n", "AUC alone", "std diff big-small"]); R["strength"] = (R["AUC alone"] - 0.5).abs()
    print("Variables ranked by how well they separate big from small INSIDE the zone (AUC above 0.5: bigger value = more likely big; below 0.5: smaller value = more likely big):")
    print(R.sort_values("strength", ascending=False).drop(columns="strength").head(14).round(3).to_string(index=False))
    # group value: leave-one-year-out, score + group columns vs score alone
    zz = z.reset_index(drop=True); lg = np.log(zz.score.clip(1e-4, 1 - 1e-4) / (1 - zz.score.clip(1e-4, 1 - 1e-4)))
    def cv_auc(cols):
        X = pd.concat([lg.rename("logit_score")] + [zz[c] for c in cols], axis=1); pred = np.zeros(len(zz))
        for Y in sorted(zz.year.unique()):
            te = (zz.year == Y).values
            mm = LGBMClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, num_leaves=8, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
            pred[te] = mm.fit(X[~te], zz.big[~te]).predict_proba(X[te])[:, 1]
        return roc_auc_score(zz.big, pred)
    base = cv_auc([]); print(f"\nIn-zone AUC with score only (same small model): {base:.3f}")
    out = []
    for g, cols in GROUPS.items():
        if cols and zz[cols].notna().any(axis=1).mean() > 0.3: out.append((g, cv_auc(cols)))
    allc = [c for cols in GROUPS.values() for c in cols]; out.append(("ALL groups together", cv_auc(allc)))
    print(pd.DataFrame(out, columns=["adding this group", "in-zone AUC"]).assign(gain=lambda t: t["in-zone AUC"] - base).round(3).sort_values("gain", ascending=False).to_string(index=False))
print("\nDone. Paste all of this output back.")
