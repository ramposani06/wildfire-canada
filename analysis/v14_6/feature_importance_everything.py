# Re-test EVERY usable column in the dataset, including the ones rejected earlier (FWI, fuel type, water/settlement distance, month, cause, ...).
# Columns that are filled in after the fire (size, out date, attack date, response, whole-fire satellite matches) are excluded: they leak the answer.
# 1) add-one test: v14.7 (22 features) + ONE extra column, on two forward splits. A column only counts if it helps on BOTH.
# 2) everything-in model vs v14.7.   3) permutation ranking of the everything-in model.
import os, re, json, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive")
CSV  = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
INFO = "final_model_v14.6_nosat_info.json"
LAT, LON, TARGET = "LATITUDE", "LONGITUDE", "is_big_fire"
P = dict(n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=31, is_unbalance=True, random_state=42, verbose=-1)
LEAK = re.compile(r"size|_ha$|^ha$|area|hect|out_date|outdate|attk|response|cfs_note|n_modis_matches|has_modis_match|year|^id$|_id$|^index|unnamed|perim|final|"
                  r"sat_zero|sensor_available|date|^fid|objectid|name|agency|fire_id|firename", re.I)
def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files: return os.path.join(root, name)
    raise FileNotFoundError(name)
base = json.load(open(find(INFO)))["features"] + [LAT, LON]
SAT = ["modis_count_early7d", "modis_max_frp_early7d", "viirs_count_early7d", "viirs_max_frp_early7d"]
df = pd.read_csv(find(CSV), low_memory=False)
df = df[df[LAT].between(41, 84) & df[LON].between(-142, -52)].dropna(subset=[TARGET]).reset_index(drop=True)
y = df[TARGET].astype(int).values; yr = df["year"].values
# month from the report date
df["rep_month"] = pd.to_datetime(df["REP_DATE"].astype(str).str[:10], errors="coerce").dt.month
df["rep_dayofyear"] = pd.to_datetime(df["REP_DATE"].astype(str).str[:10], errors="coerce").dt.dayofyear
extras, skipped = [], []
for c in df.columns:
    if c in base or c in SAT or c == TARGET: continue
    if c in ("rep_month", "rep_dayofyear"): extras.append(c); continue
    if LEAK.search(c): skipped.append(c); continue
    if df[c].dtype == object:
        if df[c].nunique() <= 40: df[c + "__code"] = df[c].astype("category").cat.codes; extras.append(c + "__code")
        else: skipped.append(c + " (text, too many values)")
    elif pd.api.types.is_numeric_dtype(df[c]):
        if df[c].notna().mean() >= 0.30 and df[c].nunique() > 1: extras.append(c)
        else: skipped.append(c + " (mostly empty or constant)")
print(f"model columns: {len(base)} | extra columns tested: {len(extras)} | skipped: {len(skipped)}")
print("EXTRA COLUMNS:", extras); print("SKIPPED:", skipped, flush=True)

def split(a, b): return (yr >= a[0]) & (yr <= a[1]), (yr >= b[0]) & (yr <= b[1])
SPLITS = [("train<=2021 -> 2022-24", split((2004, 2021), (2022, 2024))), ("train<=2024 -> 2025+", split((2004, 2024), (2025, 2026)))]
def fit_score(fs, tr, te):
    s = LGBMClassifier(**P).fit(df.loc[tr, fs], y[tr]).predict_proba(df.loc[te, fs])[:, 1]
    return roc_auc_score(y[te], s), average_precision_score(y[te], s), s
rows = {e: {} for e in extras}; basescore = {}
for name, (tr, te) in SPLITS:
    r0, p0, s0 = fit_score(base, tr, te); basescore[name] = (r0, p0)
    print(f"\n[{name}] v14.7 baseline: ROC {r0:.3f} PR {p0:.3f}  (n test {te.sum():,})", flush=True)
    for i, e in enumerate(extras):
        r1, p1, _ = fit_score(base + [e], tr, te); rows[e][name] = (r1 - r0, p1 - p0)
        if (i + 1) % 10 == 0: print(f"  {i+1}/{len(extras)} added-one tests done", flush=True)
names = [n for n, _ in SPLITS]
t = pd.DataFrame({"missing_%": [round(df[e].isna().mean() * 100, 1) for e in extras],
                  "dPR_2022-24": [rows[e][names[0]][1] for e in extras], "dPR_2025+": [rows[e][names[1]][1] for e in extras],
                  "dROC_2022-24": [rows[e][names[0]][0] for e in extras], "dROC_2025+": [rows[e][names[1]][0] for e in extras]}, index=extras)
t["both_positive"] = (t["dPR_2022-24"] > 0.003) & (t["dPR_2025+"] > 0.003)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print("\n=== Add ONE column to v14.7 (gain in PR-AUC / ROC-AUC; 'both_positive' = PR gain over +0.003 on both splits) ===")
print(t.sort_values("dPR_2025+", ascending=False).round(4).to_string())

# everything-in vs v14.7 on the main test
tr, te = SPLITS[1][1]; allf = base + extras
r1, p1, sA = fit_score(base, tr, te)[0:3]; r2, p2, sB = fit_score(allf, tr, te)
rs = np.random.RandomState(1); g = []
for _ in range(300):
    i = rs.randint(0, te.sum(), te.sum()); yy = y[te][i]
    if yy.sum(): g.append(average_precision_score(yy, sB[i]) - average_precision_score(yy, sA[i]))
print(f"\nEVERYTHING IN ({len(allf)} columns) vs v14.7 ({len(base)}), test 2025+: ROC {r2:.3f} vs {r1:.3f} | PR {p2:.3f} vs {p1:.3f} | PR gain {np.mean(g):+.3f} ({np.percentile(g,2.5):+.3f} to {np.percentile(g,97.5):+.3f})")

# permutation ranking of the everything-in model
m = LGBMClassifier(**P).fit(df.loc[tr, allf], y[tr]); Xte = df.loc[te, allf].reset_index(drop=True); yte = y[te]
b = average_precision_score(yte, m.predict_proba(Xte)[:, 1]); rp = np.random.RandomState(0); imp = {}
for f in allf:
    d = []
    for _ in range(3):
        X = Xte.copy(); X[f] = rp.permutation(X[f].values); d.append(b - average_precision_score(yte, m.predict_proba(X)[:, 1]))
    imp[f] = np.mean(d)
imp = pd.Series(imp).sort_values(ascending=False)
print("\n=== Permutation ranking of the everything-in model (PR-AUC drop when the column is shuffled), top 30 ===")
print(pd.DataFrame({"PR_drop": imp.head(30).round(4), "in_v14.7": [f in base for f in imp.head(30).index]}).to_string())
print("\nDone. Paste all of this output back.")
