# Last three checks on v14.7.
#  1) interaction columns (small, hypothesis-driven groups), added to v14.7, two forward splits
#  2) model score by final-size band, inside and outside the accessible segment (is the 100 ha line where classes overlap?)
#  3) ranking quality inside subgroups with the FROZEN model (trained to 2021, scored on 2022-26): BC, AB, human-caused, spring, near roads
import os, json, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive")
CSV  = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
INFO = "final_model_v14.6_nosat_info.json"
LAT, LON, TARGET = "LATITUDE", "LONGITUDE", "is_big_fire"
P = dict(n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=31, is_unbalance=True, random_state=42, verbose=-1)
def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files: return os.path.join(root, name)
    raise FileNotFoundError(name)
feats = json.load(open(find(INFO)))["features"] + [LAT, LON]
df = pd.read_csv(find(CSV), low_memory=False)
df = df[df[LAT].between(41, 84) & df[LON].between(-142, -52)].dropna(subset=[TARGET]).reset_index(drop=True)
y = df[TARGET].astype(int).values; yr = df["year"].values
pd.set_option("display.width", 220)

# ---------- 1) interactions
lr_, lp_ = np.log1p(df["dist_to_road_m"]), np.log1p(df["pop_within_25km"]); dry = 100 - df["relative_humidity_2m_mean_min"]
T, W, WG = df["temperature_2m_max_mean"], df["wind_speed_10m_max_mean"], df["wind_gusts_10m_max_mean"]
GROUPS = {
 "dryness x accessibility": {"road_x_dry": lr_ * dry, "road_x_temp": lr_ * T, "pop_x_dry": lp_ * dry, "pop_x_wind": lp_ * W},
 "terrain x accessibility": {"road_x_slope": lr_ * df["slope"], "pop_x_slope": lp_ * df["slope"], "road_x_elev": lr_ * df["elevation"]},
 "vegetation x accessibility": {"road_x_ndvi": lr_ * df["NDVI"], "pop_x_ndvi": lp_ * df["NDVI"]},
 "vegetation x weather": {"ndvi_x_temp": df["NDVI"] * T, "ndvi_x_dry": df["NDVI"] * dry, "ndvi_x_wind": df["NDVI"] * W},
 "terrain x weather": {"slope_x_wind": df["slope"] * W, "slope_x_gust": df["slope"] * WG, "slope_x_precip": df["slope"] * df["precipitation_sum_sum"]},
}
for g in GROUPS.values():
    for k, v in g.items(): df[k] = v
GROUPS["ALL groups together"] = {k: None for g in list(GROUPS.values()) for k in g}
SPL = [("train<=2021 -> 2022-24", yr <= 2021, (yr >= 2022) & (yr <= 2024)), ("train<=2024 -> 2025+", yr <= 2024, yr >= 2025)]
print("=== 1) Interaction columns added to v14.7 (PR-AUC gain, 95% range) ===")
base = {}
for nm, trm, tem in SPL: base[nm] = LGBMClassifier(**P).fit(df.loc[trm, feats], y[trm]).predict_proba(df.loc[tem, feats])[:, 1]
for gname, cols in GROUPS.items():
    line = f"{gname:<28}"
    for nm, trm, tem in SPL:
        fs = feats + list(cols); s = LGBMClassifier(**P).fit(df.loc[trm, fs], y[trm]).predict_proba(df.loc[tem, fs])[:, 1]; yt = y[tem]
        r = np.random.RandomState(1); g = []
        for _ in range(300):
            i = r.randint(0, len(yt), len(yt))
            if yt[i].sum(): g.append(average_precision_score(yt[i], s[i]) - average_precision_score(yt[i], base[nm][i]))
        line += f" | {nm[-7:]}: {np.mean(g):+.3f} ({np.percentile(g,2.5):+.3f} to {np.percentile(g,97.5):+.3f})"
    print(line, flush=True)

# ---------- frozen model for 2 and 3
tr = yr <= 2021; m = LGBMClassifier(**P).fit(df.loc[tr, feats], y[tr]); df["score"] = m.predict_proba(df[feats])[:, 1]
va = (yr >= 2022) & (yr <= 2024); thr = float(np.sort(df.loc[va & (y == 1), "score"].values)[int(0.35 * (va & (y == 1)).sum())])
d = df[yr >= 2022].copy(); d["big"] = d[TARGET].astype(int)
size = next((c for c in d.columns if c.upper() == "SIZE_HA"), None)
d["acc"] = d["dist_to_road_m"] <= 5000
if size:
    print(f"\n=== 2) Model score by final size band (national alert line {thr:.3f}), fires 2022-26 ===")
    sb = pd.cut(d[size], [0, 100, 500, 1000, 5000, 1e9], labels=["under 100 ha", "100-500", "500-1000", "1000-5000", "over 5000"], right=False)
    for lab, mk in (("INSIDE accessible segment (within 5 km of a road)", d.acc), ("OUTSIDE (more than 5 km from a road)", ~d.acc)):
        t = d[mk].groupby(sb[mk]).apply(lambda g: pd.Series({"fires": len(g), "median score": g.score.median(), "25th pct": g.score.quantile(.25), "75th pct": g.score.quantile(.75), "% at/above line": 100 * (g.score >= thr).mean()}))
        print("\n" + lab); print(t.round(2).to_string())

print("\n=== 3) Ranking quality inside subgroups, frozen model (train<=2021), fires 2022-26 ===")
prov = "resolved_province" if "resolved_province" in d.columns else None
d["human"] = d["CAUSE"].astype(str).str.strip().str[0].eq("H") if "CAUSE" in d.columns else False
d["mon"] = pd.to_datetime(d["REP_DATE"].astype(str).str[:10], errors="coerce").dt.month
subs = {"ALL fires": np.ones(len(d), bool), "within 5 km of a road": d.acc.values, "BC": (d[prov] == "BC").values, "AB": (d[prov] == "AB").values,
        "BC + within 5 km of road": ((d[prov] == "BC") & d.acc).values, "AB + within 5 km of road": ((d[prov] == "AB") & d.acc).values,
        "human-caused": d.human.values, "human-caused + near road": (d.human & d.acc).values, "April-May": d.mon.isin([4, 5]).values,
        "April-May + near road": (d.mon.isin([4, 5]) & d.acc).values, "Rest of Canada (not BC/AB)": (~d[prov].isin(["BC", "AB"])).values}
rows = []
for k, mk in subs.items():
    yy, ss = d.loc[mk, "big"].values, d.loc[mk, "score"].values
    if yy.sum() < 20 or yy.sum() == len(yy): rows.append((k, mk.sum(), int(yy.sum()), np.nan, np.nan, np.nan, np.nan)); continue
    rows.append((k, mk.sum(), int(yy.sum()), yy.mean(), roc_auc_score(yy, ss), average_precision_score(yy, ss), average_precision_score(yy, ss) / yy.mean()))
print(pd.DataFrame(rows, columns=["subgroup", "fires", "big fires", "base rate", "ROC-AUC", "PR-AUC", "lift"]).round(3).to_string(index=False))
print("\nDone. Paste all of this output back.")
