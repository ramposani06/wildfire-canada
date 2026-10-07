# Which data hurts the model?  Two views, main model v14.7 (22 inputs).
# A) WHERE IT IS WEAK: for the frozen model (train <=2021, alert line 0.695, test 2022-26) show each data segment's share of fires vs share of misses and false alarms.
# B) WHAT HURTS TRAINING: drop one segment from the TRAINING data (test data never changes) and see if PR-AUC goes up. Two forward splits, bootstrap ranges.
#    A segment counts as "hurting" only if removing it GAINS on both splits with a range above zero. (Many segments are tried, so a lone gain on one split is probably luck.)
# Final size is used ONLY to find fires near the 100 ha line whose big/small label is noisy; test labels are never changed.
import os, json, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive"); CSV = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
LAT, LON, TARGET = "LATITUDE", "LONGITUDE", "is_big_fire"; THR = 0.695; NB = 300
P = dict(n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=31, is_unbalance=True, random_state=42, verbose=-1)
def find(name, must=True):
    for root, _, files in os.walk(FOLDER):
        if name in files: return os.path.join(root, name)
    if must: raise FileNotFoundError(name)
feats = json.load(open(find("final_model_v14.6_nosat_info.json")))["features"] + [LAT, LON]
df = pd.read_csv(find(CSV), low_memory=False)
df = df[df[LAT].between(41, 84) & df[LON].between(-142, -52)].dropna(subset=[TARGET]); df["_row"] = df.index; df = df.reset_index(drop=True)
fp = find("province_filled_v4.csv", must=False)
if fp: df = df.merge(pd.read_csv(fp).rename(columns={"row": "_row"}), on="_row", how="left")
y = df[TARGET].astype(int).values; yr = df["year"].values
month = pd.to_datetime(df["REP_DATE"].astype(str).str[:10], errors="coerce").dt.month.values
size = next((c for c in df.columns if c.upper() == "SIZE_HA"), None); sz = df[size].values if size else np.full(len(df), np.nan)
road, pop = df["dist_to_road_m"].values, df["pop_within_25km"].values
cause = df["CAUSE"].astype(str).str.strip().str[0] if "CAUSE" in df.columns else pd.Series([""] * len(df))
seg = {"road under 1 km": road < 1000, "road 1-5 km": (road >= 1000) & (road < 5000), "road 5-20 km": (road >= 5000) & (road < 20000), "road over 20 km": road >= 20000,
       "people within 25 km: 0": pop == 0, "people 1-100": (pop > 0) & (pop <= 100), "people over 100": pop > 100,
       "Apr-May": np.isin(month, [4, 5]), "Jun-Aug": np.isin(month, [6, 7, 8]), "Sep-Oct": np.isin(month, [9, 10]), "Nov-Mar": np.isin(month, [11, 12, 1, 2, 3]),
       "cause human": (cause == "H").values, "cause natural": (cause == "N").values, "cause unknown/other": (~cause.isin(["H", "N"])).values}
if "PRESCRIBED" in df.columns: seg["prescribed burns"] = (df["PRESCRIBED"].astype(str).str.upper().str.strip() == "PB").values
if size: seg["size 50-99 ha (label noise)"] = (sz >= 50) & (sz < 100); seg["size 100-150 ha (label noise)"] = (sz >= 100) & (sz < 150); seg["size under 1 ha"] = sz < 1
if "province_filled" in df.columns:
    for p in df["province_filled"].value_counts().index[:8]: seg[f"province {p}"] = (df["province_filled"] == p).values
seg["years 2004-2011 (pre-VIIRS)"] = yr <= 2011
pd.set_option("display.width", 230); pd.set_option("display.max_rows", 200)

# ---------- A) where is the frozen model weak
tr, te = yr <= 2021, yr >= 2022
m = LGBMClassifier(**P).fit(df.loc[tr, feats], y[tr]); s = m.predict_proba(df[feats])[:, 1]
yt, st = y[te], s[te]; a = st >= THR; miss = (yt == 1) & ~a; fa = (yt == 0) & a
print(f"A) WHERE THE MODEL IS WEAK (train<=2021, test 2022-26: {te.sum():,} fires, {yt.sum():,} big, {miss.sum():,} missed, {fa.sum():,} false alarms, alert line {THR})")
rows = []
for k, mk in seg.items():
    mk = mk[te]
    if mk.sum() < 300 or yt[mk].sum() < 30: continue
    yy, ss = yt[mk], st[mk]; aa = a[mk]
    rows.append((k, mk.sum(), 100 * mk.mean(), 100 * yy.mean(), roc_auc_score(yy, ss), average_precision_score(yy, ss), 100 * (aa & (yy == 1)).sum() / yy.sum(),
                 100 * (aa & (yy == 1)).sum() / max(aa.sum(), 1), 100 * miss[mk].sum() / miss.sum(), 100 * fa[mk].sum() / fa.sum()))
A = pd.DataFrame(rows, columns=["segment", "fires", "% of fires", "% big", "ROC-AUC", "PR-AUC", "recall %", "precision %", "% of ALL misses", "% of ALL false alarms"])
A[["ROC-AUC", "PR-AUC"]] = A[["ROC-AUC", "PR-AUC"]].round(3); print(A.round(1).to_string(index=False))
print("   (a segment with a bigger share of misses than of fires is where big fires slip through)")

# ---------- B) does removing a segment from training help?
rng = np.random.default_rng(0)
def split(name, trm, tem):
    yte = y[tem]; base = LGBMClassifier(**P).fit(df.loc[trm, feats], y[trm]).predict_proba(df.loc[tem, feats])[:, 1]; out = {}
    for k, mk in seg.items():
        keep = trm & ~mk
        if (trm & mk).sum() < 300 or (trm & mk).sum() > 0.6 * trm.sum(): continue
        s2 = LGBMClassifier(**P).fit(df.loc[keep, feats], y[keep]).predict_proba(df.loc[tem, feats])[:, 1]; d = []
        for _ in range(NB):
            b = rng.integers(0, len(yte), len(yte))
            if yte[b].sum(): d.append(average_precision_score(yte[b], s2[b]) - average_precision_score(yte[b], base[b]))
        out[k] = (int((trm & mk).sum()), np.mean(d), np.percentile(d, 2.5), np.percentile(d, 97.5))
    print(f"   {name}: baseline PR-AUC {average_precision_score(yte, base):.3f} ({tem.sum():,} test fires) done", flush=True); return out
print("\nB) REMOVE ONE SEGMENT FROM TRAINING (test unchanged). PR-AUC gain from removing it:")
R1 = split("train<=2021 -> 2022-24", yr <= 2021, (yr >= 2022) & (yr <= 2024)); R2 = split("train<=2024 -> 2025+", yr <= 2024, yr >= 2025)
rows = []
for k in R1:
    if k in R2: rows.append((k, R1[k][0], f"{R1[k][1]:+.3f} ({R1[k][2]:+.3f} to {R1[k][3]:+.3f})", f"{R2[k][1]:+.3f} ({R2[k][2]:+.3f} to {R2[k][3]:+.3f})", "HURTS" if R1[k][2] > 0 and R2[k][2] > 0 else ("helps" if R1[k][3] < 0 and R2[k][3] < 0 else "")))
print(pd.DataFrame(rows, columns=["segment removed from training", "train fires removed (<=2021)", "gain 2022-24", "gain 2025+", "verdict"]).to_string(index=False))
print("\nDone. Paste all of this output back.")
