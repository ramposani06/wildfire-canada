# Use only the VIIRS era (fires from 2012 on) for the whole model, and compare with training on 2004+.
# Test fires are the same in both setups, so the numbers are comparable.  Two forward splits:
#   train 2012-2021 (or 2004-2021) -> test 2022-24;   train 2012-2024 (or 2004-2024) -> test 2025+
# Feature sets: v14.7 (22, no satellite);  + VIIRS columns only (24);  + all 4 satellite columns (26).  Satellite = reference only (report-day timing risk).
import os, json, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive")
CSV  = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
INFO = "final_model_v14.6_nosat_info.json"
LAT, LON, TARGET = "LATITUDE", "LONGITUDE", "is_big_fire"
VI = ["viirs_count_early7d", "viirs_max_frp_early7d"]; MO = ["modis_count_early7d", "modis_max_frp_early7d"]
P = dict(n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=31, is_unbalance=True, random_state=42, verbose=-1)
def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files: return os.path.join(root, name)
    raise FileNotFoundError(name)
base = json.load(open(find(INFO)))["features"] + [LAT, LON]
SETS = {"22 features (no satellite)": base, "22 + VIIRS (2 cols)": base + VI, "22 + VIIRS + MODIS (4 cols)": base + VI + MO}
df = pd.read_csv(find(CSV), low_memory=False)
df = df[df[LAT].between(41, 84) & df[LON].between(-142, -52)].dropna(subset=[TARGET]).reset_index(drop=True)
y = df[TARGET].astype(int).values; yr = df["year"].values
print("Fires and big-fire rate per training start:")
for a in (2004, 2012): m = (yr >= a) & (yr <= 2021); print(f"  {a}-2021: {m.sum():,} fires, {y[m].mean():.1%} big")
print("Satellite columns empty (share) by period:", {f"{a}-{b}": round(float(df.loc[(yr >= a) & (yr <= b), VI + MO].isna().mean().mean()), 3) for a, b in ((2004, 2011), (2012, 2021), (2022, 2026))}, flush=True)
SPL = [("train->2021, test 2022-24", 2021, (yr >= 2022) & (yr <= 2024)), ("train->2024, test 2025+  ", 2024, yr >= 2025)]
for sn, last, tem in SPL:
    print(f"\n######## {sn}  (test n={tem.sum():,}, {y[tem].mean():.1%} big) ########")
    ref = None
    for fname, fs in SETS.items():
        res = {}
        for start in (2004, 2012):
            trm = (yr >= start) & (yr <= last)
            res[start] = LGBMClassifier(**P).fit(df.loc[trm, fs], y[trm]).predict_proba(df.loc[tem, fs])[:, 1]
        yt = y[tem]
        if ref is None: ref = res[2004]            # the main model (22 features, trained from 2004)
        for start in (2004, 2012):
            s = res[start]; r = np.random.RandomState(1); g = []
            for _ in range(300):
                i = r.randint(0, len(yt), len(yt))
                if yt[i].sum(): g.append(average_precision_score(yt[i], s[i]) - average_precision_score(yt[i], ref[i]))
            o = np.argsort(-s); top = yt[o[:int(.15 * len(s))]].sum() / yt.sum()
            print(f"  {fname:<28} trained from {start}: ROC {roc_auc_score(yt, s):.3f} | PR {average_precision_score(yt, s):.3f} | top15% {top:.0%} | PR vs main model {np.mean(g):+.3f} ({np.percentile(g,2.5):+.3f} to {np.percentile(g,97.5):+.3f})", flush=True)
print("\nDone. Paste all of this output back.")
