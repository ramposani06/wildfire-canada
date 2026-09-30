# Compare v14.6 with simple baselines on 2025 fires that have fire weather index (FWI) columns.
# FWI exists for 2012-2025 only. Baselines train on <=2024 fires that have FWI. Run in Colab.
import os, json, joblib, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
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
model = joblib.load(find("final_model_v14.6_clean.pkl"))
feats = info["features"]
FWI = [c for c in ["FWI_mean_7d", "FWI_max_7d", "DC_mean_7d", "DC_max_7d", "BUI_mean_7d",
                   "ISI_max_7d", "FFMC_mean_7d", "DC_mean_30d", "BUI_mean_30d"] if c in df.columns]
WEATHER = [f for f in feats if f.startswith(("temperature", "precipitation", "wind", "relative_humidity", "sunshine"))]
has = df[FWI].notna().any(axis=1)
train, test = df[(df[YEAR] <= 2024) & has], df[(df[YEAR] == 2025) & has].copy()
y = test[T].values
print("Train", len(train), "| Test (2025)", len(test), "| base rate", round(y.mean() * 100, 1), "%")

def top(y, s, frac=.15):
    k = int(len(s) * frac); return y[np.argsort(-s)[:k]].sum() / y.sum()

def row(name, s):
    return {"score": name, "ROC-AUC": round(roc_auc_score(y, s), 3),
            "PR-AUC": round(average_precision_score(y, s), 3), "top 15%": f"{top(y, s)*100:.0f}%"}

def lr(cols):
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))
    return m.fit(train[cols], train[T]).predict_proba(test[cols])[:, 1]

rows = [row("Model v14.6", model.predict_proba(test[feats])[:, 1]),
        row("FWI columns only (logistic)", lr(FWI)), row("Weather columns only (logistic)", lr(WEATHER))]
rows += [row(f"{c} alone", test[c].fillna(test[c].median()).values) for c in FWI]
print(pd.DataFrame(rows).to_string(index=False))
