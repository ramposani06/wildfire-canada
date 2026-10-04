# Two-step check on TWO splits with bootstrap ranges.
# Split A: train <=2018, alert lines on 2019-21, test 2022-24.   Split B: train <=2021, alert lines on 2022-24, test 2025+.
import os, json, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive"); CSV = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
SAT = ["modis_count_early7d", "modis_max_frp_early7d", "viirs_count_early7d", "viirs_max_frp_early7d"]
LAT, LON, TARGET = "LATITUDE", "LONGITUDE", "is_big_fire"; NB = 500
P = dict(n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=31, is_unbalance=True, random_state=42, verbose=-1)
def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files: return os.path.join(root, name)
    raise FileNotFoundError(name)
f1 = json.load(open(find("final_model_v14.6_nosat_info.json")))["features"] + [LAT, LON]; f2 = f1 + SAT
df = pd.read_csv(find(CSV), low_memory=False)
df = df[df[LAT].between(41, 84) & df[LON].between(-142, -52)].dropna(subset=[TARGET]).reset_index(drop=True)
y = df[TARGET].astype(int).values; yr = df["year"].values
def thr65(s, yy): return float(np.sort(s[yy == 1])[int(0.35 * (yy == 1).sum())])
def ci(v): return f"{np.mean(v):+.3f} ({np.percentile(v,2.5):+.3f} to {np.percentile(v,97.5):+.3f})"
rng = np.random.default_rng(0)
def run(name, trm, cal, tst):
    m1 = LGBMClassifier(**P).fit(df.loc[trm, f1], y[trm]); m2 = LGBMClassifier(**P).fit(df.loc[trm, f2], y[trm])
    s1, s2 = m1.predict_proba(df[f1])[:, 1], m2.predict_proba(df[f2])[:, 1]
    t1, t2 = thr65(s1[cal], y[cal]), thr65(s2[cal], y[cal])
    ix = np.where(tst)[0]; yt = y[ix]; a1, a2 = s1[ix] >= t1, s2[ix] >= t2; o = a1 | a2; S1, S2 = s1[ix], s2[ix]
    print(f"\n######## {name}: {len(ix):,} fires, {yt.sum():,} big ########")
    D = {k: [] for k in ["prauc2-1", "rec s1", "rec s2", "rec or", "or-s1 rec", "or-s1 flag", "prec s1", "prec s2", "prec or"]}
    for _ in range(NB):
        b = rng.integers(0, len(ix), len(ix)); yb = yt[b]
        if yb.sum() == 0: continue
        D["prauc2-1"].append(average_precision_score(yb, S2[b]) - average_precision_score(yb, S1[b]))
        r = lambda a: (a[b] & (yb == 1)).sum() / yb.sum(); pr = lambda a: (a[b] & (yb == 1)).sum() / max(a[b].sum(), 1)
        D["rec s1"].append(r(a1)); D["rec s2"].append(r(a2)); D["rec or"].append(r(o)); D["or-s1 rec"].append(r(o) - r(a1)); D["or-s1 flag"].append(o[b].mean() - a1[b].mean())
        D["prec s1"].append(pr(a1)); D["prec s2"].append(pr(a2)); D["prec or"].append(pr(o))
    print(f"alert lines: step 1 {t1:.3f}, step 2 {t2:.3f}")
    print(f"ROC step1 {roc_auc_score(yt,S1):.3f} step2 {roc_auc_score(yt,S2):.3f} | PR step1 {average_precision_score(yt,S1):.3f} step2 {average_precision_score(yt,S2):.3f}")
    print("PR gain step 2 over step 1:", ci(D["prauc2-1"]))
    for k, lab in [("rec s1", "recall step 1"), ("rec s2", "recall step 2"), ("rec or", "recall OR"), ("prec s1", "precision step 1"), ("prec s2", "precision step 2"), ("prec or", "precision OR")]:
        print(f"{lab:18s}", ci(D[k]))
    print("OR minus step 1: recall", ci(D["or-s1 rec"]), "| extra fires flagged", ci(D["or-s1 flag"]))
    miss = (~a1) & (yt == 1); print(f"step 1 missed {miss.sum():,}; step 2 catches {(miss & a2).sum():,} ({(miss & a2).sum()/max(miss.sum(),1):.0%})")
run("Split A (train<=2018, lines 2019-21, test 2022-24)", yr <= 2018, (yr >= 2019) & (yr <= 2021), (yr >= 2022) & (yr <= 2024))
run("Split B (train<=2021, lines 2022-24, test 2025+)", yr <= 2021, (yr >= 2022) & (yr <= 2024), yr >= 2025)
print("\nDone. Paste all of this output back.")
