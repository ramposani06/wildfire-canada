# Does using satellite passes through the END OF THE NEXT DAY (report day + 1) help more than through the end of the report day?
# Both versions are REBUILT from the detection database with the same code, so the comparison is fair.
#   same-day: detections 0..7 days before the report date (what step 2 uses)      next-day: detections from 1 day AFTER the report date back to 7 days before
# Valid only after the next day's passes are in (a 1-2 day delay). Splits and bootstrap are the same as two_stage_validate.py.
import os, json, sqlite3, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.neighbors import BallTree
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive"); CSV = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
SAT = ["modis_count_early7d", "modis_max_frp_early7d", "viirs_count_early7d", "viirs_max_frp_early7d"]
LAT, LON, TARGET = "LATITUDE", "LONGITUDE", "is_big_fire"; NB = 500; R_KM = 10.0
P = dict(n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=31, is_unbalance=True, random_state=42, verbose=-1)
def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files: return os.path.join(root, name)
    raise FileNotFoundError(name)
f1 = json.load(open(find("final_model_v14.6_nosat_info.json")))["features"] + [LAT, LON]
df = pd.read_csv(find(CSV), low_memory=False)
df = df[df[LAT].between(41, 84) & df[LON].between(-142, -52)].dropna(subset=[TARGET]).reset_index(drop=True)
y = df[TARGET].astype(int).values; yr = df["year"].values
rep = pd.to_datetime(df["REP_DATE"].astype(str).str[:10], errors="coerce"); rd = rep.values.astype("datetime64[D]").astype("int64"); ok = rep.notna().values
con = sqlite3.connect(find("wildfire_corrected.db"))
def load(t):
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({t})")]; frp = next((c for c in cols if "frp" in c.lower()), None)
    d = pd.read_sql(f"SELECT latitude, longitude, acq_date{', ' + frp + ' AS frp' if frp else ''} FROM {t}", con)
    d["day"] = pd.to_datetime(d["acq_date"].astype(str).str[:10], errors="coerce").values.astype("datetime64[D]").astype("int64")
    if "frp" not in d: d["frp"] = np.nan
    return d.dropna(subset=["latitude", "longitude"])
fire_rad = np.radians(df[[LAT, LON]].values)
def build(kmin):  # window: k = report_day - detection_day in [kmin, 7]; kmin=0 same-day, kmin=-1 includes next day
    out = {}
    for s, t in [("modis", "modis_detections"), ("viirs", "viirs_detections")]:
        d = load(t); print(f"{s}: {len(d):,} detections, {pd.to_datetime(d['day'].min(), unit='D').date()} to {pd.to_datetime(d['day'].max(), unit='D').date()}") if kmin == 0 else None
        tree = BallTree(np.radians(d[["latitude", "longitude"]].values), metric="haversine"); dd, ff = d["day"].values, d["frp"].values
        cnt, mx = np.zeros(len(df)), np.zeros(len(df))
        for i, idx in enumerate(tree.query_radius(fire_rad, r=R_KM / 6371.0)):
            if len(idx) == 0 or not ok[i]: continue
            k = rd[i] - dd[idx]; m = (k >= kmin) & (k <= 7)
            if m.any(): cnt[i] = m.sum(); v = ff[idx[m]]; mx[i] = np.nanmax(v) if np.isfinite(v).any() else 0.0
        out[f"{s}_count_early7d"], out[f"{s}_max_frp_early7d"] = cnt, mx
    return pd.DataFrame(out)
S0, S1 = build(0), build(-1)
print("sanity: rebuilt same-day vs dataset columns, correlation:", {c: round(float(np.corrcoef(S0[c], df[c].fillna(0))[0, 1]), 3) for c in SAT})
for c in SAT: df["s0_" + c] = S0[c].values; df["s1_" + c] = S1[c].values
f_same, f_next = f1 + ["s0_" + c for c in SAT], f1 + ["s1_" + c for c in SAT]
def thr65(s, yy): return float(np.sort(s[yy == 1])[int(0.35 * (yy == 1).sum())])
def ci(v): return f"{np.mean(v):+.3f} ({np.percentile(v,2.5):+.3f} to {np.percentile(v,97.5):+.3f})"
rng = np.random.default_rng(0)
def run(name, trm, cal, tst):
    ms = {k: LGBMClassifier(**P).fit(df.loc[trm, f], y[trm]) for k, f in [("step1", f1), ("same", f_same), ("next", f_next)]}
    sc = {k: m.predict_proba(df[{"step1": f1, "same": f_same, "next": f_next}[k]])[:, 1] for k, m in ms.items()}
    th = {k: thr65(sc[k][cal], y[cal]) for k in sc}; ix = np.where(tst)[0]; yt = y[ix]; S = {k: sc[k][ix] for k in sc}; A = {k: S[k] >= th[k] for k in S}
    print(f"\n######## {name}: {len(ix):,} fires, {yt.sum():,} big ########")
    for k in S: print(f"{k:6s} ROC {roc_auc_score(yt,S[k]):.3f} PR {average_precision_score(yt,S[k]):.3f} | recall {(A[k]&(yt==1)).sum()/yt.sum():.1%} flagged {A[k].mean():.1%} precision {(A[k]&(yt==1)).sum()/max(A[k].sum(),1):.1%}")
    g = {"same-day over step 1": ("same", "step1"), "next-day over step 1": ("next", "step1"), "next-day over same-day": ("next", "same")}
    D = {k: [] for k in g}
    for _ in range(NB):
        b = rng.integers(0, len(ix), len(ix)); yb = yt[b]
        if yb.sum() == 0: continue
        pr = {k: average_precision_score(yb, S[k][b]) for k in S}
        for kk, (a, c) in g.items(): D[kk].append(pr[a] - pr[c])
    for kk in g: print(f"PR gain, {kk:24s}", ci(D[kk]))
    o = A["same"] | A["next"]; miss = (~A["step1"]) & (yt == 1)
    print(f"step 1 missed {miss.sum():,}: same-day catches {(miss & A['same']).sum()}, next-day catches {(miss & A['next']).sum()}")
run("Split A (train<=2018, lines 2019-21, test 2022-24)", yr <= 2018, (yr >= 2019) & (yr <= 2021), (yr >= 2022) & (yr <= 2024))
run("Split B (train<=2021, lines 2022-24, test 2025+)", yr <= 2021, (yr >= 2022) & (yr <= 2024), yr >= 2025)
print("\nDone. Paste all of this output back.")
