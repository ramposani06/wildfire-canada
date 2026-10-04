# Show how real fires are scored today. Models trained to 2024 only; examples are real 2025 fires the model never saw.
# Step 1 = v14.7 at report time (22 inputs, alert line 0.695). Step 2 = same + 4 satellite columns, re-score after the day's passes (alert line 0.770).
import os, json, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
warnings.filterwarnings("ignore")
FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive"); CSV = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
LAT, LON, TARGET = "LATITUDE", "LONGITUDE", "is_big_fire"
P = dict(n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=31, is_unbalance=True, random_state=42, verbose=-1)
def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files: return os.path.join(root, name)
    raise FileNotFoundError(name)
i1 = json.load(open(find("final_model_v14.7_info.json"))); i2 = json.load(open(find("final_model_v14.8_stage2_info.json")))
f1, f2 = i1["features"], i2["features"]; t1, t2 = i1["threshold_recall"], i2["threshold_recall"]; c1, c2 = i1["calibrator"], i2["calibrator"]
df = pd.read_csv(find(CSV), low_memory=False)
df = df[df[LAT].between(41, 84) & df[LON].between(-142, -52)].dropna(subset=[TARGET]).reset_index(drop=True)
y = df[TARGET].astype(int).values; yr = df["year"].values; tr = yr <= 2024
m1 = LGBMClassifier(**P).fit(df.loc[tr, f1], y[tr]); m2 = LGBMClassifier(**P).fit(df.loc[tr, f2], y[tr])
d = df[yr == 2025].copy(); d["s1"] = m1.predict_proba(d[f1])[:, 1]; d["s2"] = m2.predict_proba(d[f2])[:, 1]
chance = lambda s, c: 1 / (1 + np.exp(-(c["coef"] * np.log(np.clip(s, 1e-4, 1 - 1e-4) / (1 - np.clip(s, 1e-4, 1 - 1e-4))) + c["intercept"])))
d["chance1"] = chance(d["s1"], c1); d["a1"] = d["s1"] >= t1; d["a2"] = d["s2"] >= t2
size = next((c for c in d.columns if c.upper() == "SIZE_HA"), None); prov = next((c for c in ("resolved_province", "PROVINCE", "SRC_AGENCY") if c in d.columns), None)
d["big"] = d[TARGET].astype(int)
groups = [("CAUGHT BIG FIRE (step 1 alerted, it was big)", (d.big == 1) & d.a1), ("MISSED BIG FIRE (step 1 quiet, it was big)", (d.big == 1) & ~d.a1),
          ("FALSE ALARM (step 1 alerted, it stayed small)", (d.big == 0) & d.a1), ("QUIET SMALL FIRE (no alert, stayed small)", (d.big == 0) & ~d.a1)]
print(f"2025 fires: {len(d):,}. Step 1 alert line {t1:.3f}; step 2 alert line {t2:.3f}. 'Chance' = calibrated probability of a big fire (>100 ha).\n")
for title, mk in groups:
    g = d[mk].sample(2, random_state=3)
    print("=" * 100 + f"\n{title}")
    for _, r in g.iterrows():
        print(f"\n  Reported {str(r['REP_DATE'])[:10]} | province {r[prov] if prov else '?'} | lat {r[LAT]:.2f}, lon {r[LON]:.2f}")
        print(f"  Inputs known at report time: road {r['dist_to_road_m']/1000:.1f} km away | people within 25 km {r['pop_within_25km']:.0f} | NDVI {r['NDVI']:.2f} | slope {r['slope']:.1f} | "
              f"humidity (7-day min) {r['relative_humidity_2m_mean_min']:.0f}% | max temp avg {r['temperature_2m_max_mean']:.1f} C")
        print(f"  STEP 1 (report time): score {r['s1']:.3f} -> chance {r['chance1']:.0%} -> {'ALERT' if r['a1'] else 'no alert'}")
        print(f"  Satellite by end of report day: MODIS {r['modis_count_early7d']:.0f} hits, VIIRS {r['viirs_count_early7d']:.0f} hits, strongest heat (VIIRS FRP) {r['viirs_max_frp_early7d']:.1f}")
        print(f"  STEP 2 (after satellite): score {r['s2']:.3f} -> {'ALERT' if r['a2'] else 'no alert'}")
        print(f"  WHAT HAPPENED: {'BIG fire' if r['big'] else 'stayed small'}" + (f", final size {r[size]:,.0f} ha" if size else ""))
print("\nDone. Paste all of this output back.")
