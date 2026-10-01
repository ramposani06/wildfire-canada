# Do roads and population stand in for how fires are managed?
# TEST 1: inside each province (train <=2024, test 2025+): gain from roads/people.
# TEST 2: 2004-2011 fires that have a PROTZONE name (train 2004-09, test 2010-11).
# PROTZONE is empty from 2012 on, so it cannot be tested on 2025-26. Run in Colab.
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
RP = ["dist_to_road_m", "pop_within_10km", "pop_within_25km"]
no_rp = [f for f in feats if f not in RP]

def top15(y, s): k = max(int(len(s) * .15), 1); return y[np.argsort(-s)[:k]].sum() / max(y.sum(), 1)

# TEST 1
train, test = df[df[YEAR] <= 2024], df[df[YEAR] >= 2025].copy()
test["s_all"] = clone(base).fit(train[feats], train[T]).predict_proba(test[feats])[:, 1]
test["s_no"] = clone(base).fit(train[no_rp], train[T]).predict_proba(test[no_rp])[:, 1]
rows = []
for p, g in test.groupby("resolved_province"):
    if len(g) >= 200 and g[T].sum() >= 20:
        rows.append({"province": p, "fires": len(g), "big": int(g[T].sum()),
                     "model all": round(roc_auc_score(g[T], g["s_all"]), 3),
                     "model without roads/people": round(roc_auc_score(g[T], g["s_no"]), 3),
                     "road dist alone": round(roc_auc_score(g[T], g["dist_to_road_m"].fillna(g["dist_to_road_m"].median())), 3),
                     "pop 25km alone (inverted)": round(roc_auc_score(g[T], -g["pop_within_25km"].fillna(0)), 3)})
r1 = pd.DataFrame(rows); r1["gain from roads/people"] = (r1["model all"] - r1["model without roads/people"]).round(3)
print("TEST 1 - ROC-AUC inside each province, 2025+"); print(r1.to_string(index=False))

# TEST 2
z = df["PROTZONE"].astype("string").str.strip()
d2 = df[(df[YEAR] <= 2011) & z.notna() & (z != "")].copy()
d2["PZ"] = z[d2.index].astype("category")
tr2, te2 = d2[d2[YEAR] <= 2009], d2[d2[YEAR] >= 2010]
print(f"\nTEST 2 - train {len(tr2)} (2004-09, {int(tr2[T].sum())} big) | test {len(te2)} (2010-11, {int(te2[T].sum())} big)")
y2 = te2[T].values
def run2(name, cols):
    s = clone(base).fit(tr2[cols], tr2[T]).predict_proba(te2[cols])[:, 1]
    return {"setup": name, "ROC-AUC": round(roc_auc_score(y2, s), 3),
            "PR-AUC": round(average_precision_score(y2, s), 3), "top 15%": f"{top15(y2, s)*100:.0f}%"}
print(pd.DataFrame([run2("All 24", feats), run2("All 24 + zone", feats + ["PZ"]),
                    run2("Without roads/people", no_rp), run2("Without roads/people + zone", no_rp + ["PZ"]),
                    run2("Zone only", ["PZ"]), run2("Roads/people only", RP)]).to_string(index=False))
