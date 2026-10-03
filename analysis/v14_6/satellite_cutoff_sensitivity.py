# How much of the satellite benefit survives if the fire is scored at a given LOCAL TIME on the report day?
# Report times are not in the data, so instead of guessing one, we show the whole curve:
# only detections that happened before local hour H on the report day (plus all earlier days) are visible.
# H=0 means no report-day detections; H=24 means the whole report day (what the full model was trained on).
# Detection time is UTC; local time is approximated by solar time (UTC + longitude/15 hours).
import os, sqlite3, json, warnings, joblib
import numpy as np, pandas as pd
from sklearn.neighbors import BallTree
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")

FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive")
CSV    = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
MODEL, INFO, DB = "final_model_v14.6_clean.pkl", "final_model_v14.6_clean_info.json", "wildfire_corrected.db"
CUTOFFS = [0, 3, 6, 9, 12, 15, 18, 21, 24]
RADIUS_KM, THRESH = 10.0, 0.770
def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files: return os.path.join(root, name)
    raise FileNotFoundError(name)

df = pd.read_csv(find(CSV), low_memory=False)
df = df[df["LATITUDE"].between(41, 84) & df["LONGITUDE"].between(-142, -52)].dropna(subset=["is_big_fire"])
df["_d"] = pd.to_datetime(df["REP_DATE"].astype(str).str[:10], errors="coerce")
df = df[(df["year"] >= 2025) & df["_d"].between("2025-01-01", "2026-12-31")].reset_index(drop=True)
y = df["is_big_fire"].astype(int).values
print(f"Test fires 2025+: {len(df):,}")

con = sqlite3.connect(find(DB))
def load(table):
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
    frp = next((c for c in cols if "frp" in c.lower()), None)
    q = f"SELECT latitude, longitude, acq_date, acq_time{', ' + frp + ' AS frp' if frp else ''} FROM {table} WHERE acq_date >= '2024-12-01'"
    d = pd.read_sql(q, con)
    d["acq_date"] = pd.to_datetime(d["acq_date"].astype(str).str[:10], errors="coerce")
    t = pd.to_numeric(d["acq_time"], errors="coerce").fillna(0).astype(int)
    d["utc_h"] = (t // 100) + (t % 100) / 60.0
    if "frp" not in d: d["frp"] = np.nan
    d["loc_h_total"] = d["acq_date"].values.astype("datetime64[D]").astype(int) * 24 + d["utc_h"] + d["longitude"] / 15.0
    d["loc_day"] = np.floor(d["loc_h_total"] / 24).astype(int)
    d["loc_hour"] = d["loc_h_total"] - d["loc_day"] * 24
    return d.dropna(subset=["acq_date"])
sens = {"modis": load("modis_detections"), "viirs": load("viirs_detections")}
print({k: len(v) for k, v in sens.items()}, "detections loaded since 2024-12")

fire_rad = np.radians(df[["LATITUDE", "LONGITUDE"]].values)
rep_day = df["_d"].values.astype("datetime64[D]").astype(int)
feat = {(s, c): (np.zeros(len(df)), np.zeros(len(df))) for s in sens for c in CUTOFFS}
first_hours = []
for s, d in sens.items():
    tree = BallTree(np.radians(d[["latitude", "longitude"]].values), metric="haversine")
    ld, lh, fr = d["loc_day"].values, d["loc_hour"].values, d["frp"].values
    for i, idx in enumerate(tree.query_radius(fire_rad, r=RADIUS_KM / 6371.0)):
        if len(idx) == 0: continue
        k = rep_day[i] - ld[idx]
        inwin = (k >= 0) & (k <= 7)
        if not inwin.any(): continue
        ii, kk, hh = idx[inwin], k[inwin], lh[idx[inwin]]
        sameday = kk == 0
        if sameday.any(): first_hours.append(hh[sameday].min())
        for c in CUTOFFS:
            vis = ~sameday | (hh < c)
            if vis.any():
                feat[(s, c)][0][i] = vis.sum()
                f = fr[ii[vis]]; feat[(s, c)][1][i] = np.nanmax(f) if not np.all(np.isnan(f)) else 0

fh = np.array(first_hours)
if len(fh):
    print(f"\nFires with a report-day detection: {len(fh):,}. Local solar hour of the EARLIEST report-day detection: "
          f"10th pct {np.percentile(fh,10):.0f}h, median {np.median(fh):.0f}h, 90th pct {np.percentile(fh,90):.0f}h")

obj = joblib.load(find(MODEL)); model = obj.get("model", obj) if isinstance(obj, dict) else obj
feats = json.load(open(find(INFO)))["features"]
def score(frame): return model.predict_proba(frame[feats])[:, 1]
def line(label, s):
    fl = s >= THRESH; top = y[np.argsort(-s)[:int(.15 * len(s))]].sum() / y.sum()
    return f"{label:<40} ROC-AUC {roc_auc_score(y, s):.3f} | PR-AUC {average_precision_score(y, s):.3f} | 0.770 rule catches {(fl & (y==1)).sum()/y.sum():.0%} | top 15% catches {top:.0%}"
print("\n== Scores on 2025+ when only detections before local hour H on the report day are visible ==")
print(line("Stored features (as reported)", score(df)))
zero = df.copy()
for c in ("modis_count_early7d", "modis_max_frp_early7d", "viirs_count_early7d", "viirs_max_frp_early7d"): zero[c] = 0.0
print(line("No satellite (all 4 set to 0)", score(zero)))
for c in CUTOFFS:
    f = df.copy()
    for s in sens: f[f"{s}_count_early7d"], f[f"{s}_max_frp_early7d"] = feat[(s, c)]
    tag = " (no report-day detections)" if c == 0 else (" (whole report day)" if c == 24 else "")
    print(line(f"Rebuilt, visible before {c:>2}:00 local{tag}", score(f)))
print("\nHow to read: this is the full model (trained with the whole report day) scored with less information. "
      "The later the fire is reported in the day, the more of the satellite benefit is real. "
      "The rebuilt rows use only the detections stored in the database, which is about half of those behind the stored features.")
print("Done. Paste all of this output back.")
