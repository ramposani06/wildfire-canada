# Do ensembles beat the single LightGBM? (no-satellite main model, 20 features, v4 data)
# Models: LightGBM (the current one), LightGBM with bagging, XGBoost, CatBoost, sklearn HistGradientBoosting.
# The ensemble is a plain average of ranks (equal weights, nothing tuned on the test years).
# Check 1: train <=2021, score 2022-2024 (validation). Check 2: train <=2024, score 2025+ (forward test).
import os, json, subprocess, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: import catboost
except ImportError: subprocess.run([sys.executable, "-m", "pip", "install", "-q", "catboost"]); import catboost
import xgboost as xgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import rankdata

FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive")
CSV    = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
INFO   = "final_model_v14.6_nosat_info.json"
TARGET, YEAR, LAT, LON = "is_big_fire", "year", "LATITUDE", "LONGITUDE"
def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files: return os.path.join(root, name)
    raise FileNotFoundError(name)

df = pd.read_csv(find(CSV), low_memory=False)
df = df[df[LAT].between(41, 84) & df[LON].between(-142, -52)].dropna(subset=[TARGET]).reset_index(drop=True)
feats = json.load(open(find(INFO)))["features"]
y = df[TARGET].astype(int).values; yr = df[YEAR].values
X = df[feats]

def models(pos_weight):
    return {
        "LightGBM (current)": LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=31, is_unbalance=True, random_state=42, verbose=-1),
        "LightGBM bagged": LGBMClassifier(n_estimators=400, learning_rate=0.03, max_depth=6, num_leaves=31, subsample=0.7, subsample_freq=1,
                                          colsample_bytree=0.7, is_unbalance=True, random_state=7, verbose=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                                     scale_pos_weight=pos_weight, tree_method="hist", random_state=42, verbosity=0),
        "CatBoost": CatBoostClassifier(iterations=400, learning_rate=0.05, depth=6, auto_class_weights="Balanced", random_seed=42, verbose=0),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=7, class_weight="balanced", random_state=42),
    }

def run(train, test, label):
    ytr, yte = y[train], y[test]
    scores = {}
    for name, m in models((ytr == 0).sum() / max((ytr == 1).sum(), 1)).items():
        scores[name] = m.fit(X[train], ytr).predict_proba(X[test])[:, 1]
    rk = {k: rankdata(v) / len(v) for k, v in scores.items()}
    scores["Ensemble: all 5 (rank average)"] = np.mean(list(rk.values()), axis=0)
    scores["Ensemble: LightGBM + XGBoost + CatBoost"] = np.mean([rk["LightGBM (current)"], rk["XGBoost"], rk["CatBoost"]], axis=0)
    print(f"\n== {label}: train {train.sum():,} | test {test.sum():,} ({yte.mean():.3f} big) ==")
    for k, s in scores.items():
        top = yte[np.argsort(-s)[:int(.15 * len(s))]].sum() / yte.sum()
        print(f"{k:<42} ROC-AUC {roc_auc_score(yte, s):.3f} | PR-AUC {average_precision_score(yte, s):.3f} | top 15% catches {top:.0%}")
    base = scores["LightGBM (current)"]; r = np.random.RandomState(1)
    for k in ("Ensemble: all 5 (rank average)", "Ensemble: LightGBM + XGBoost + CatBoost"):
        d = []
        for _ in range(500):
            i = r.randint(0, len(yte), len(yte))
            if yte[i].sum() == 0: continue
            d.append(average_precision_score(yte[i], scores[k][i]) - average_precision_score(yte[i], base[i]))
        print(f"  gain in PR-AUC, {k}: {np.mean(d):+.3f} (95% {np.percentile(d,2.5):+.3f} to {np.percentile(d,97.5):+.3f})")

run(yr <= 2021, (yr >= 2022) & (yr <= 2024), "VALIDATION 2022-2024")
run(yr <= 2024, yr >= 2025, "FORWARD TEST 2025+")
print("\nHow to read: the forward test is the one to report. Trust a gain only if its 95% range stays above zero.")
print("Done. Paste all of this output back.")
