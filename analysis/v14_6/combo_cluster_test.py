# Do combinations of the weak (weather/terrain) features hide big fires the model misses?  Three honest checks.
# Model = v14.7 trained to 2021. Discovery = 2022-24 (look for patterns here). Confirm = 2025+ (the pattern must still hold here).
#  A) rules: a shallow tree fit on what the model gets wrong, using only the weak features -> readable "if X and Y" rules, checked on 2025+
#  B) missed big fires: cluster them and see what they have in common, then see if the same groups appear in 2025+
#  C) cluster id as a new feature: k-means on the weak features, add the cluster (and distance to centre) to v14.7, forward test
import os, json, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive")
CSV  = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
INFO = "final_model_v14.6_nosat_info.json"
LAT, LON, TARGET, EPS = "LATITUDE", "LONGITUDE", "is_big_fire", 1e-4
P = dict(n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=31, is_unbalance=True, random_state=42, verbose=-1)
def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files: return os.path.join(root, name)
    raise FileNotFoundError(name)
base = json.load(open(find(INFO)))["features"]; feats = base + [LAT, LON]
STRONG = ["dist_to_road_m", "pop_within_10km", "pop_within_25km", "province_encoded", "NDVI", LAT, LON]
WEAK = [f for f in feats if f not in STRONG]            # 13 weather + elevation + slope
df = pd.read_csv(find(CSV), low_memory=False)
df = df[df[LAT].between(41, 84) & df[LON].between(-142, -52)].dropna(subset=[TARGET]).reset_index(drop=True)
y = df[TARGET].astype(int).values; yr = df["year"].values
tr, va, te = yr <= 2021, (yr >= 2022) & (yr <= 2024), yr >= 2025
print(f"weak features ({len(WEAK)}): {WEAK}")
m = LGBMClassifier(**P).fit(df.loc[tr, feats], y[tr])
raw = m.predict_proba(df[feats])[:, 1]
pv = np.clip(raw[va], EPS, 1 - EPS); lr = LogisticRegression(C=1e6).fit(np.log(pv / (1 - pv)).reshape(-1, 1), y[va])
p = lr.predict_proba(np.log(np.clip(raw, EPS, 1 - EPS) / (1 - np.clip(raw, EPS, 1 - EPS))).reshape(-1, 1))[:, 1]
resid = y - p
print(f"discovery 2022-24: n={va.sum():,} | confirm 2025+: n={te.sum():,}")

print("\n=== A) Rules from the weak features on what the model gets wrong (residual = actual - model chance) ===")
Xw = df[WEAK].fillna(df[WEAK].median())
tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=400, random_state=0).fit(Xw[va], resid[va])
print(export_text(tree, feature_names=WEAK, decimals=2))
leaf_va, leaf_te = tree.apply(Xw[va]), tree.apply(Xw[te])
rows = []
for L in np.unique(leaf_va):
    a, b = leaf_va == L, leaf_te == L
    rows.append((L, a.sum(), resid[va][a].mean(), b.sum(), resid[te][b].mean() if b.sum() else np.nan,
                 y[va][a].mean(), p[va][a].mean(), y[te][b].mean() if b.sum() else np.nan, p[te][b].mean() if b.sum() else np.nan))
r = pd.DataFrame(rows, columns=["leaf", "n_2022-24", "miss_2022-24", "n_2025+", "miss_2025+", "actual_2022-24", "model_2022-24", "actual_2025+", "model_2025+"]).set_index("leaf")
pd.set_option("display.width", 250); print("miss = actual big-fire rate minus the model's average chance (positive = model too low)")
print(r.round(3).sort_values("miss_2022-24", ascending=False).to_string())

print("\n=== B) Big fires the model misses (score under the 65%-recall threshold), 2022-24 vs caught ===")
sv = raw[va]; thr = float(np.sort(sv[y[va] == 1])[int(0.35 * (y[va] == 1).sum())])
sc = StandardScaler().fit(df.loc[tr, feats]); Z = pd.DataFrame(sc.transform(df[feats]), columns=feats)
big = y == 1; miss = big & (raw < thr); caught = big & (raw >= thr)
for nm, mk in (("2022-24", va), ("2025+", te)):
    print(f"{nm}: big fires {int((big & mk).sum()):,} | missed {int((miss & mk).sum()):,} ({(miss & mk).sum()/(big & mk).sum():.0%}) | threshold {thr:.3f}")
d = (Z[miss & va].mean() - Z[caught & va].mean()).sort_values(); print("Missed vs caught big fires (2022-24), difference in std-devs; biggest gaps:")
print(pd.concat([d.head(4), d.tail(4)]).round(2).to_string())
km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(Z.loc[miss & va, feats])
allbig_mean = Z[big & va].mean()
for k in range(4):
    mk = km.labels_ == k; prof = (Z.loc[miss & va, feats][mk].mean() - allbig_mean).sort_values(key=abs, ascending=False).head(4)
    lab_te = km.predict(Z.loc[miss & te, feats]); print(f"  group {k}: {mk.sum()} missed fires in 2022-24 ({mk.mean():.0%}); in 2025+ {int((lab_te==k).sum())} of {len(lab_te)} ({(lab_te==k).mean():.0%}) | stands out: " + ", ".join(f"{i} {v:+.1f}" for i, v in prof.items()))

print("\n=== C) Cluster id of the weak features as a new column (forward tests) ===")
def run(trm, tem, name):
    r0 = None; out = []
    s0 = LGBMClassifier(**P).fit(df.loc[trm, feats], y[trm]).predict_proba(df.loc[tem, feats])[:, 1]
    b = (roc_auc_score(y[tem], s0), average_precision_score(y[tem], s0))
    for k in (8, 20, 50):
        sc2 = StandardScaler().fit(df.loc[trm, WEAK]); kmk = MiniBatchKMeans(n_clusters=k, n_init=5, random_state=0, batch_size=4096).fit(sc2.transform(df.loc[trm, WEAK]))
        Zall = sc2.transform(df[WEAK]); cl = kmk.predict(Zall); dist = np.min(kmk.transform(Zall), axis=1)
        X = df[feats].copy(); X["wk_cluster"] = cl; X["wk_dist"] = dist
        s1 = LGBMClassifier(**P).fit(X.loc[trm], y[trm]).predict_proba(X.loc[tem])[:, 1]
        a = (roc_auc_score(y[tem], s1), average_precision_score(y[tem], s1))
        rs = np.random.RandomState(1); g = []
        for _ in range(300):
            i = rs.randint(0, tem.sum(), tem.sum()); yy = y[tem][i]
            if yy.sum(): g.append(average_precision_score(yy, s1[i]) - average_precision_score(yy, s0[i]))
        print(f"  {name} k={k:<3} ROC {b[0]:.3f}->{a[0]:.3f} | PR {b[1]:.3f}->{a[1]:.3f} | PR gain {np.mean(g):+.3f} ({np.percentile(g,2.5):+.3f} to {np.percentile(g,97.5):+.3f})", flush=True)
run(yr <= 2021, va, "train<=2021 -> 2022-24")
run(yr <= 2024, te, "train<=2024 -> 2025+   ")
print("\nDone. Paste all of this output back.")
