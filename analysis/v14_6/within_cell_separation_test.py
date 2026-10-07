# Inside each road x people cell: does ANY of the 22 report-time inputs separate big fires from small ones, and is there information the global model fails to use?
# Part 1 (descriptive, all years 2004-26): per cell, each input's single-column AUC (0.5 = no separation) and standardized effect (big minus small, in std units).
# Part 2 (predictive): per cell, a small model trained ONLY on that cell's fires <=2021 and tested on that cell's fires 2022-26, compared with the national model (train <=2021)
#   on the SAME test fires. If the cell model cannot beat the national score, the cell has no extra usable information in these inputs.
import os, json, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive"); CSV = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
LAT, LON, TARGET = "LATITUDE", "LONGITUDE", "is_big_fire"; NB = 300
P = dict(n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=31, is_unbalance=True, random_state=42, verbose=-1)
PC = dict(n_estimators=150, learning_rate=0.05, max_depth=4, num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, is_unbalance=True, random_state=1, verbose=-1)
def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files: return os.path.join(root, name)
    raise FileNotFoundError(name)
feats = json.load(open(find("final_model_v14.6_nosat_info.json")))["features"] + [LAT, LON]
df = pd.read_csv(find(CSV), low_memory=False)
df = df[df[LAT].between(41, 84) & df[LON].between(-142, -52)].dropna(subset=[TARGET]).reset_index(drop=True)
y = df[TARGET].astype(int).values; yr = df["year"].values; road, pop = df["dist_to_road_m"].values, df["pop_within_25km"].values
rb = np.select([road < 1000, road < 5000, road < 20000], ["road<1km", "road1-5km", "road5-20km"], "road>20km"); pb = np.select([pop == 0, pop <= 100], ["pop0", "pop1-100"], "pop>100")
cell = np.char.add(np.char.add(rb.astype(str), "|"), pb.astype(str))
s_nat = LGBMClassifier(**P).fit(df.loc[yr <= 2021, feats], y[yr <= 2021]).predict_proba(df[feats])[:, 1]
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200); rng = np.random.default_rng(0)
cells = [c for c in sorted(set(cell)) if (cell == c).sum() >= 1500 and y[cell == c].sum() >= 100]
# only the cells that matter for misses are shown in detail, but all qualifying cells get part 2
print(f"cells analysed: {len(cells)}\n\nPART 1: best single separators inside each cell (all years). AUC>0.5: bigger value = more likely big.")
summary = []
for c in cells:
    m = cell == c; X, yy = df.loc[m, feats], y[m]; rows = []
    for f in feats:
        v = X[f].values; ok = ~np.isnan(v)
        if ok.sum() < 200 or len(np.unique(v[ok])) < 3: continue
        a = roc_auc_score(yy[ok], v[ok]); sd = np.nanstd(v[ok]); d = (np.nanmean(v[ok][yy[ok] == 1]) - np.nanmean(v[ok][yy[ok] == 0])) / sd if sd else np.nan
        rows.append((f, a, d))
    R = pd.DataFrame(rows, columns=["input", "AUC alone", "effect (std)"]); R["sep"] = (R["AUC alone"] - 0.5).abs(); R = R.sort_values("sep", ascending=False)
    print(f"\n--- {c}: {m.sum():,} fires, {yy.mean():.1%} big. Best 5 of {len(R)} inputs:"); print(R.head(5).drop(columns="sep").round(3).to_string(index=False))
    summary.append((c, int(m.sum()), yy.mean(), R.iloc[0]["input"], R.iloc[0]["AUC alone"], float((R["sep"] > 0.1).sum())))

print("\n\nPART 2: a model trained only inside the cell (<=2021) vs the national model, tested on the cell's 2022-26 fires")
rows = []
for c in cells:
    trm, tem = (cell == c) & (yr <= 2021), (cell == c) & (yr >= 2022)
    if trm.sum() < 500 or y[trm].sum() < 40 or y[tem].sum() < 40: continue
    sc = LGBMClassifier(**PC).fit(df.loc[trm, feats], y[trm]).predict_proba(df.loc[tem, feats])[:, 1]; sn = s_nat[tem]; yt = y[tem]; d = []
    for _ in range(NB):
        b = rng.integers(0, len(yt), len(yt))
        if yt[b].sum(): d.append(roc_auc_score(yt[b], sc[b]) - roc_auc_score(yt[b], sn[b]))
    rows.append((c, int(tem.sum()), int(yt.sum()), f"{roc_auc_score(yt, sn):.3f}", f"{roc_auc_score(yt, sc):.3f}", f"{np.mean(d):+.3f} ({np.percentile(d, 2.5):+.3f} to {np.percentile(d, 97.5):+.3f})",
                 f"{average_precision_score(yt, sn):.3f}", f"{average_precision_score(yt, sc):.3f}", "cell model BETTER" if np.percentile(d, 2.5) > 0 else ("national BETTER" if np.percentile(d, 97.5) < 0 else "no difference")))
print(pd.DataFrame(rows, columns=["cell", "test fires", "big", "ROC national", "ROC cell model", "ROC gain of cell model", "PR national", "PR cell model", "verdict"]).to_string(index=False))
print("\nDone. Paste all of this output back.")
