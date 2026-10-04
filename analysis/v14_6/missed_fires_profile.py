# What do the MISSED big fires look like?  Descriptive only (final size is used to describe fires AFTER the fact, never as a model input).
# Model v14.7 trained to 2021, threshold for 65% recall from 2022-24. Looks at 2022-24 and 2025+ together (both are out-of-sample for this model).
import os, json, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
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
tr, va = yr <= 2021, (yr >= 2022) & (yr <= 2024)
m = LGBMClassifier(**P).fit(df.loc[tr, feats], y[tr]); df["score"] = m.predict_proba(df[feats])[:, 1]
thr = float(np.sort(df.loc[va & (y == 1), "score"].values)[int(0.35 * (va & (y == 1)).sum())])
d = df[yr >= 2022].copy(); d["big"] = d[TARGET].astype(int)
d["month"] = pd.to_datetime(d["REP_DATE"].astype(str).str[:10], errors="coerce").dt.month
d["alert"] = d["score"] >= thr
d["group"] = np.select([d.big.eq(1) & d.alert, d.big.eq(1) & ~d.alert, d.big.eq(0) & d.alert], ["caught big", "MISSED big", "false alarm"], "quiet small")
size = next((c for c in d.columns if c.upper() == "SIZE_HA"), None)
prov = next((c for c in ("resolved_province", "PROVINCE", "SRC_AGENCY") if c in d.columns), None)
print(f"threshold {thr:.3f} | fires 2022-26 {len(d):,} | big {int(d.big.sum()):,} | caught {(d.group=='caught big').sum():,} | MISSED {(d.group=='MISSED big').sum():,} | false alarms {(d.group=='false alarm').sum():,}")
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 100)
B = d[d.big == 1]

print("\n1) How wrong was the model? Score of missed big fires (threshold is %.3f):" % thr)
ms = d[d.group == "MISSED big"]["score"]
bands = pd.cut(ms, [-1, 0.05, 0.2, 0.4, 0.55, thr], labels=["under 0.05 (very sure it was small)", "0.05-0.2", "0.2-0.4", "0.4-0.55", f"0.55-{thr:.2f} (near miss)"])
print(bands.value_counts(normalize=True).sort_index().mul(100).round(0).astype(int).astype(str).add("%").to_string())

if size:
    print("\n2) Catch rate by final size (bigger fires should be easier):")
    sb = pd.cut(B[size], [100, 200, 500, 1000, 5000, 1e9], labels=["100-200 ha", "200-500", "500-1000", "1000-5000", "over 5000"], right=False)
    print(B.groupby(sb).apply(lambda g: pd.Series({"big fires": len(g), "caught %": round(100 * g.alert.mean())})).to_string())

if prov:
    print(f"\n3) By {prov} (provinces with 40+ big fires): where are fires missed?")
    g = B.groupby(prov).apply(lambda g: pd.Series({"big fires": len(g), "missed": int((~g.alert).sum()), "missed %": round(100 * (~g.alert).mean()),
                                                   "share of all misses %": round(100 * (~g.alert).sum() / (B.alert == False).sum())}))
    print(g[g["big fires"] >= 40].sort_values("missed", ascending=False).to_string())

print("\n4) By month of report:")
g = B.groupby("month").apply(lambda g: pd.Series({"big fires": len(g), "missed %": round(100 * (~g.alert).mean())})); print(g[g["big fires"] >= 30].T.to_string())

if "CAUSE" in d.columns:
    d["cause"] = d["CAUSE"].astype(str).str.strip().str[0].map({"N": "natural", "H": "human"}).fillna("unknown/other"); B = d[d.big == 1]
    print("\n5) By cause:"); print(B.groupby("cause").apply(lambda g: pd.Series({"big fires": len(g), "missed %": round(100 * (~g.alert).mean())})).to_string())

print("\n6) By distance to nearest road:")
rb = pd.cut(B["dist_to_road_m"], [-1, 1000, 5000, 20000, 1e9], labels=["under 1 km", "1-5 km", "5-20 km", "over 20 km"])
print(B.groupby(rb).apply(lambda g: pd.Series({"big fires": len(g), "missed %": round(100 * (~g.alert).mean())})).to_string())

print("\n7) Average values: missed big vs caught big vs false alarm vs quiet small")
cols = ["score", "LATITUDE", "dist_to_road_m", "pop_within_25km", "NDVI", "slope", "temperature_2m_max_mean", "relative_humidity_2m_mean_min", "precipitation_sum_sum", "wind_speed_10m_max_mean"]
print(d.groupby("group")[[c for c in cols if c in d.columns]].median().round(2).T.to_string())

print("\n8) Twelve missed big fires (the ones with the lowest scores):")
ex = d[d.group == "MISSED big"].nsmallest(12, "score")
show = [c for c in (prov, "year", "month", size, "score", "dist_to_road_m", "pop_within_25km") if c and c in ex.columns]
print(ex[show].round(3).to_string(index=False))
print("\nDone. Paste all of this output back.")
