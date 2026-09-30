# Refit v14.6 with one feature group removed at a time, then each group alone.
# Train on <=2024, test on 2025+. Run in Colab after mounting Drive.
import os, json, joblib, warnings
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")

FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive")
CSV = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v3.csv")
T, YEAR = "is_big_fire", "year"

def find(name):
    for root, _, fs in os.walk(FOLDER):
        if name in fs: return os.path.join(root, name)
    raise FileNotFoundError(name)

df = pd.read_csv(find(CSV), low_memory=False)
info = json.load(open(find("final_model_v14.6_clean_info.json")))
base = joblib.load(find("final_model_v14.6_clean.pkl"))
feats = info["features"]
groups = {
 "Weather": [f for f in feats if f.startswith(("temperature", "precipitation", "wind", "relative_humidity", "sunshine"))],
 "Terrain": [f for f in ["elevation", "slope", "NDVI"] if f in feats],
 "Satellite": [f for f in feats if f.startswith(("modis", "viirs"))],
 "Roads/people": [f for f in ["dist_to_road_m", "pop_within_10km", "pop_within_25km"] if f in feats],
 "Province": [f for f in ["province_encoded"] if f in feats],
}
assert sorted(sum(groups.values(), [])) == sorted(feats), "groups do not cover all features"
train, test = df[df[YEAR] <= 2024], df[df[YEAR] >= 2025]
y = test[T].values

def top(y, s, frac=.15):
    k = int(len(s) * frac); return y[np.argsort(-s)[:k]].sum() / y.sum()

def run(name, cols):
    m = clone(base).fit(train[cols], train[T]); s = m.predict_proba(test[cols])[:, 1]
    return {"setup": name, "features": len(cols), "ROC-AUC": round(roc_auc_score(y, s), 3),
            "PR-AUC": round(average_precision_score(y, s), 3), "top 15%": f"{top(y, s)*100:.0f}%"}

rows = [run("All features", feats)]
rows += [run(f"Without {g}", [f for f in feats if f not in c]) for g, c in groups.items()]
rows += [run(f"Only {g}", c) for g, c in groups.items()]
print(f"Train {len(train)} (<=2024) | Test {len(test)} (2025+) | base rate {y.mean()*100:.1f}%")
print(pd.DataFrame(rows).to_string(index=False))
