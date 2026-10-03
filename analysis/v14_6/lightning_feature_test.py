# Does lightning in the days BEFORE the report day help the no-satellite model?
# Data: CanCPLD cloud-to-ground flashes, 0.1 degree, 3-hourly, 2004-2024 (cg_YYYY.nc on Drive).
# Only whole UTC days strictly before the report date are used, so nothing from the report day leaks in.
# Test: train 2004-2021, test 2022-2024 (the lightning data ends in 2024). Same model with and without 5 lightning columns.
import os, json, glob, warnings
import numpy as np, pandas as pd, xarray as xr
from scipy.ndimage import uniform_filter
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")

FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive")
CSV    = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
INFO   = "final_model_v14.6_nosat_info.json"
CGDIR  = os.path.join(FOLDER, "lightning_project", "cancpld_cg")
OUT    = os.path.join(FOLDER, "lightning_features_for_fires.csv")
NEW = ["lt3_prev1", "lt3_prev3", "lt3_prev7", "lt21_prev3", "lt21_prev7"]   # 3x3 cells (~0.3 deg) and 21x21 cells (~2 deg)
TARGET, LAT, LON = "is_big_fire", "LATITUDE", "LONGITUDE"
def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files: return os.path.join(root, name)
    raise FileNotFoundError(name)

df = pd.read_csv(find(CSV), low_memory=False)
df = df[df[LAT].between(41, 84) & df[LON].between(-142, -52)].dropna(subset=[TARGET]).copy()
df["_d"] = pd.to_datetime(df["REP_DATE"].astype(str).str[:10], errors="coerce")
df["_id"] = df.index.astype(str)
df = df[df["year"].between(2004, 2024) & df["_d"].notna()].copy()
print(f"fires 2004-2024: {len(df):,}")

if os.path.exists(OUT):
    feats_df = pd.read_csv(OUT); feats_df["_id"] = feats_df["_id"].astype(str)
else:
    rows = []
    for yr in range(2004, 2025):
        f = os.path.join(CGDIR, f"cg_{yr}.nc")
        sub = df[df["year"] == yr]
        if not os.path.exists(f) or sub.empty: continue
        ds = xr.open_dataset(f); lat, lon = ds["lat"].values, ds["lon"].values
        days = ds["time"].values.astype("datetime64[D]"); ud = np.unique(days)
        cg = ds["cg_flashes"]
        D = np.zeros((len(ud), len(lat), len(lon)), dtype=np.float32)
        for j, d in enumerate(ud):
            D[j] = np.nan_to_num(cg.isel(time=np.where(days == d)[0]).sum("time").values)
        S3 = np.stack([uniform_filter(D[j], size=3, mode="constant") * 9 for j in range(len(ud))])
        S21 = np.stack([uniform_filter(D[j], size=21, mode="constant") * 441 for j in range(len(ud))])
        C3 = np.concatenate([np.zeros((1,) + S3.shape[1:], np.float32), np.cumsum(S3, 0)]); del S3
        C21 = np.concatenate([np.zeros((1,) + S21.shape[1:], np.float32), np.cumsum(S21, 0)]); del S21, D
        iy = np.round((lat[0] - sub[LAT].values) / (lat[0] - lat[1])).astype(int)
        ix = np.round((sub[LON].values - lon[0]) / (lon[1] - lon[0])).astype(int)
        ok = (iy >= 0) & (iy < len(lat)) & (ix >= 0) & (ix < len(lon))
        iy, ix = np.clip(iy, 0, len(lat) - 1), np.clip(ix, 0, len(lon) - 1)
        di = (sub["_d"].values.astype("datetime64[D]") - ud[0]).astype(int)    # index of the report day in this year
        def win(C, n):      # sum over the n whole days strictly before the report day
            hi = np.clip(di, 0, len(ud)); lo = np.clip(di - n, 0, len(ud))
            return C[hi, iy, ix] - C[lo, iy, ix]
        res = pd.DataFrame({"_id": sub["_id"].values, "lt3_prev1": win(C3, 1), "lt3_prev3": win(C3, 3), "lt3_prev7": win(C3, 7),
                            "lt21_prev3": win(C21, 3), "lt21_prev7": win(C21, 7)})
        res.loc[~ok, NEW] = np.nan
        rows.append(res); print(f"{yr}: {len(sub):,} fires done", flush=True)
        del C3, C21
    feats_df = pd.concat(rows); feats_df.to_csv(OUT, index=False)
m = df.merge(feats_df, on="_id", how="inner")
print(f"\nfires with lightning features: {len(m):,}")
print("Big-fire rate by lightning in the 7 days before (3x3 cells), 2022-2024 fires:")
t = m[m["year"] >= 2022]; print(t.groupby(pd.cut(t["lt3_prev7"], [-1, 0, 5, 50, 1e9], labels=["0", "1-5", "6-50", "50+"]))[TARGET].agg(["size", "mean"]).round(3).to_string())

feats = json.load(open(find(INFO)))["features"]
PARAMS = dict(n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=31, is_unbalance=True, random_state=42, verbose=-1)
tr, te = m[m["year"] <= 2021], m[m["year"] >= 2022]
ytr, yte = tr[TARGET].astype(int).values, te[TARGET].astype(int).values
sA = LGBMClassifier(**PARAMS).fit(tr[feats], ytr).predict_proba(te[feats])[:, 1]
sB = LGBMClassifier(**PARAMS).fit(tr[feats + NEW], ytr).predict_proba(te[feats + NEW])[:, 1]
nat = (te["CAUSE"].astype(str).str.strip().str[0] == "N").values if "CAUSE" in te else np.zeros(len(te), bool)
def boot(mask):
    r = np.random.RandomState(1); a, p = [], []
    yy, A, B = yte[mask], sA[mask], sB[mask]
    for _ in range(500):
        i = r.randint(0, len(yy), len(yy))
        if yy[i].sum() == 0: continue
        a.append(roc_auc_score(yy[i], B[i]) - roc_auc_score(yy[i], A[i])); p.append(average_precision_score(yy[i], B[i]) - average_precision_score(yy[i], A[i]))
    return np.mean(a), np.percentile(a, [2.5, 97.5]), np.mean(p), np.percentile(p, [2.5, 97.5])
print(f"\ntrain 2004-2021 ({len(tr):,}) | test 2022-2024 ({len(te):,}, {yte.mean():.3f} big)")
for lab, mask in (("All test fires", np.ones(len(te), bool)), ("Natural-cause fires only", nat)):
    if mask.sum() < 200: continue
    yy = yte[mask]
    for nm, s in (("without lightning", sA), ("with lightning", sB)):
        print(f"{lab:<26} {nm:<18} ROC-AUC {roc_auc_score(yy, s[mask]):.3f} | PR-AUC {average_precision_score(yy, s[mask]):.3f}  (n={mask.sum():,})")
    a, ar, p, pr = boot(mask)
    print(f"{'':<26} gain: ROC-AUC {a:+.3f} ({ar[0]:+.3f} to {ar[1]:+.3f}) | PR-AUC {p:+.3f} ({pr[0]:+.3f} to {pr[1]:+.3f})")
print("\nDone. Paste all of this output back.")
