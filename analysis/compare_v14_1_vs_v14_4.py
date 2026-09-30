# Wildfire project - compare older model (v14.1) with v14.4
# Run in Colab after mounting Drive.

import os, joblib
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

# ---------- CONFIG ----------
FOLDER = "/content/drive/MyDrive"
CSV    = "unified_dataset_2004_2026_FINAL_v2.csv"
MODELS = {"v14.1": "final_model_v14_1.pkl",
          "v14.4": "final_model_v14.4.pkl"}
TARGET = "is_big_fire"
YEAR   = "year"
# ----------------------------

def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError(name)

def get_feats(obj):
    model = obj.get("model", obj) if isinstance(obj, dict) else obj
    feats = None
    if isinstance(obj, dict):
        feats = obj.get("features") or obj.get("feature_names")
    if feats is None:
        feats = getattr(model, "feature_names_in_", None)
    if feats is None:
        cc = getattr(model, "calibrated_classifiers_", None)
        est = getattr(cc[0], "estimator", None) if cc else model
        feats = getattr(est, "feature_name_", None)
        if feats is None and hasattr(est, "booster_"):
            feats = est.booster_.feature_name()
    if feats is None:
        raise ValueError("Cannot read feature names - tell me the model type: " + type(model).__name__)
    return model, [str(f) for f in feats]

df = pd.read_csv(find(CSV), low_memory=False)
loaded = {}
for label, fname in MODELS.items():
    model, feats = get_feats(joblib.load(find(fname)))
    loaded[label] = (model, feats)
    print(f"{label}: {type(model).__name__}, {len(feats)} features")

# ---- 1. feature difference ----
a, b = loaded["v14.1"][1], loaded["v14.4"][1]
print("\n== Feature difference ==")
print("Only in v14.4:", sorted(set(b) - set(a)))
print("Only in v14.1:", sorted(set(a) - set(b)))
missing = {k: [f for f in v[1] if f not in df.columns] for k, v in loaded.items()}
print("Columns missing from the CSV:", missing)

# ---- 2. score both on the SAME rows ----
def score(label, mask, name):
    t = df[mask].dropna(subset=[TARGET])
    model, feats = loaded[label]
    p = model.predict_proba(t[feats])[:, 1]
    y = t[TARGET].values
    return name, len(t), roc_auc_score(y, p), average_precision_score(y, p)

slices = [("2022", df[YEAR] == 2022), ("2023", df[YEAR] == 2023), ("2024", df[YEAR] == 2024),
          ("2025", df[YEAR] == 2025), ("2026", df[YEAR] == 2026), ("2025+", df[YEAR] >= 2025)]
print("\n== Same rows, both models ==")
print(f"{'slice':6s} {'rows':>6s} | {'v14.1 AUC':>9s} {'v14.1 PR':>8s} | {'v14.4 AUC':>9s} {'v14.4 PR':>8s} | PR change")
for name, mask in slices:
    _, n, a1, p1 = score("v14.1", mask, name)
    _, _, a4, p4 = score("v14.4", mask, name)
    print(f"{name:6s} {n:>6,d} | {a1:>9.3f} {p1:>8.3f} | {a4:>9.3f} {p4:>8.3f} | {p4 - p1:+.3f}")
print("\nDone. Paste all of this output back.")
