# Two checks for the no-satellite model (train 2004-2021):
#  1) Does exact location (latitude, longitude) help? Scored on 2022-24 and on 2025+.
#  2) How predictable are other size cutoffs (>10, >100, >500, >1000 ha)? Same 20 features, lift = PR-AUC / base rate.
import os, json, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive")
CSV  = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
INFO = "final_model_v14.6_nosat_info.json"
TARGET, LAT, LON = "is_big_fire", "LATITUDE", "LONGITUDE"
def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files: return os.path.join(root, name)
    raise FileNotFoundError(name)
df = pd.read_csv(find(CSV), low_memory=False)
df = df[df[LAT].between(41, 84) & df[LON].between(-142, -52)].dropna(subset=[TARGET]).copy()
feats = json.load(open(find(INFO)))["features"]
P = dict(n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=31, is_unbalance=True, random_state=42, verbose=-1)
tr = df[df["year"].between(2004, 2021)]
def boot(y, A, B, n=300):
    r = np.random.RandomState(1); g = []
    for _ in range(n):
        i = r.randint(0, len(y), len(y))
        if y[i].sum() == 0: continue
        g.append(average_precision_score(y[i], B[i]) - average_precision_score(y[i], A[i]))
    return np.mean(g), np.percentile(g, [2.5, 97.5])

print("=== 1) Exact location ===")
ytr = tr[TARGET].astype(int).values
mA = LGBMClassifier(**P).fit(tr[feats], ytr)
mB = LGBMClassifier(**P).fit(tr[feats + [LAT, LON]], ytr)
for lab, te in (("2022-24", df[df["year"].between(2022, 2024)]), ("2025+", df[df["year"] >= 2025])):
    y = te[TARGET].astype(int).values
    A, B = mA.predict_proba(te[feats])[:, 1], mB.predict_proba(te[feats + [LAT, LON]])[:, 1]
    g, ci = boot(y, A, B)
    print(f"{lab:<8} n={len(te):,} | without: ROC {roc_auc_score(y, A):.3f} PR {average_precision_score(y, A):.3f} | with lat/lon: ROC {roc_auc_score(y, B):.3f} PR {average_precision_score(y, B):.3f} | PR gain {g:+.3f} ({ci[0]:+.3f} to {ci[1]:+.3f})")

print("\n=== 2) Other size cutoffs (test 2025+, same 20 features) ===")
size = next((c for c in df.columns if c.upper() in ("SIZE_HA", "FINAL_SIZE_HA", "SIZE")), None)
if size is None:
    print("No size column found. Columns:", [c for c in df.columns if "SIZE" in c.upper() or "HA" in c.upper()])
else:
    te = df[df["year"] >= 2025]
    print(f"{'cutoff':<10}{'big share':>10}{'ROC-AUC':>9}{'PR-AUC':>8}{'lift':>7}")
    for cut in (10, 100, 500, 1000):
        ytr_c = (tr[size] > cut).astype(int).values; yte_c = (te[size] > cut).astype(int).values
        s = LGBMClassifier(**P).fit(tr[feats], ytr_c).predict_proba(te[feats])[:, 1]
        base = yte_c.mean(); pr = average_precision_score(yte_c, s)
        print(f">{cut:<8} ha{base:>9.3f}{roc_auc_score(yte_c, s):>9.3f}{pr:>8.3f}{pr/base:>6.1f}x")
print("\nDone. Paste all of this output back.")
