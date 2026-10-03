# Does adding CAUSE (N natural / H human / U unknown) raise the score of the no-satellite model?
# Caveat: cause may be set AFTER an investigation, so this is an UPPER BOUND until timing is confirmed.
import os, json, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")

FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive")
CSV    = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
INFO   = "final_model_v14.6_nosat_info.json"
TARGET, YEAR, LAT, LON = "is_big_fire", "year", "LATITUDE", "LONGITUDE"
PARAMS = dict(n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=31, is_unbalance=True, random_state=42, verbose=-1)
def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files: return os.path.join(root, name)
    raise FileNotFoundError(name)

df = pd.read_csv(find(CSV), low_memory=False)
df = df[df[LAT].between(41, 84) & df[LON].between(-142, -52)].dropna(subset=[TARGET]).reset_index(drop=True)
cc = next((c for c in ("CAUSE", "CAUSE2", "cause") if c in df.columns), None)
if cc is None: raise SystemExit("No CAUSE column in the unified file. Columns: " + ", ".join(df.columns))
df["cause_code"] = df[cc].astype(str).str.strip().str[0].map({"N": 0, "H": 1, "U": 2})
print(f"cause column: {cc}")
print("Share with a cause by year (2018+):", df[df[YEAR] >= 2018].groupby(YEAR)["cause_code"].apply(lambda s: f"{s.notna().mean():.0%}").to_dict())
print("Big-fire rate by cause, 2025+:", df[df[YEAR] >= 2025].groupby("cause_code")[TARGET].agg(["size", "mean"]).round(3).to_dict("index"))
print("Share of cause values by year (N/H/U), 2022-2026:")
print(df[df[YEAR] >= 2022].groupby(YEAR)["cause_code"].value_counts(normalize=True).unstack().round(2).to_string())

feats = json.load(open(find(INFO)))["features"]
y = df[TARGET].astype(int).values; yr = df[YEAR].values
tr, te = yr <= 2024, yr >= 2025
def run(cols, label):
    s = LGBMClassifier(**PARAMS).fit(df.loc[tr, cols], y[tr]).predict_proba(df.loc[te, cols])[:, 1]
    return s
sA, sB = run(feats, "A"), run(feats + ["cause_code"], "B")
yt = y[te]; r = np.random.RandomState(1); dA, dP = [], []
for _ in range(500):
    i = r.randint(0, len(yt), len(yt))
    dA.append(roc_auc_score(yt[i], sB[i]) - roc_auc_score(yt[i], sA[i]))
    dP.append(average_precision_score(yt[i], sB[i]) - average_precision_score(yt[i], sA[i]))
print(f"\nTrain <=2024 ({tr.sum():,}), test 2025+ ({te.sum():,}, {yt.mean():.3f} big)")
for lab, s in (("No-satellite model (20 features)", sA), ("+ cause (21 features)", sB)):
    top = yt[np.argsort(-s)[:int(.15 * len(s))]].sum() / yt.sum()
    print(f"{lab:<36} ROC-AUC {roc_auc_score(yt, s):.3f} | PR-AUC {average_precision_score(yt, s):.3f} | top 15% catches {top:.0%}")
print(f"Gain in ROC-AUC {np.mean(dA):+.3f} (95% {np.percentile(dA,2.5):+.3f} to {np.percentile(dA,97.5):+.3f})")
print(f"Gain in PR-AUC  {np.mean(dP):+.3f} (95% {np.percentile(dP,2.5):+.3f} to {np.percentile(dP,97.5):+.3f})")
print("\nDone. Paste all of this output back.")
