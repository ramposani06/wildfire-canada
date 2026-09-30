# Wildfire project - export SMALL summary tables for the monitoring dashboard.
# Run in Colab after mounting Drive. Writes monitoring_summary.json (no raw rows, safe to share).
import os, json, warnings, joblib
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive")
CSV    = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v3.csv")
MODEL, INFO, CAL = ("final_model_v14.6_clean.pkl", "final_model_v14.6_clean_info.json",
                    "final_model_v14.6_calibrator.json")
TARGET, YEAR = "is_big_fire", "year"
REF = (2022, 2024)          # reference period for drift
EPS = 1e-4

def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files: return os.path.join(root, name)
    raise FileNotFoundError(name)

df = pd.read_csv(find(CSV), low_memory=False)
info = json.load(open(find(INFO))); cal = json.load(open(find(CAL)))
feats, thr = info["features"], float(info["threshold_recall"])
model = joblib.load(find(MODEL))

# month: use REP_DATE (flexible parse), fall back to MONTH columns
d = pd.to_datetime(df["REP_DATE"].astype(str).str[:10], errors="coerce")
df["_month"] = d.dt.month
df = df[df[YEAR] >= 2022].copy()
df["raw"] = model.predict_proba(df[feats])[:, 1]
p = np.clip(df["raw"], EPS, 1 - EPS)
df["cal"] = 1 / (1 + np.exp(-(cal["coef"] * np.log(p / (1 - p)) + cal["intercept"])))
df["alert"] = df["raw"] >= thr

def period(y): return "2022-24" if y <= 2024 else str(int(y))
df["period"] = df[YEAR].map(period)

out = {"threshold": thr, "note": "2022-24 scores come from a model trained through 2024 (in-sample); 2025+ is forward."}
m = df.groupby([YEAR, "_month"]).agg(n=("raw", "size"), alert_rate=("alert", "mean"),
        mean_raw=("raw", "mean"), mean_cal=("cal", "mean"), big_rate=(TARGET, "mean")).reset_index()
out["monthly"] = m.round(4).rename(columns={YEAR: "year", "_month": "month"}).to_dict("records")

bins = np.array([0, .05, .1, .2, .3, .5, .7, 1.0001]); rel = []
for y in (2025, 2026):
    s = df[df[YEAR] == y]
    g = s.groupby(pd.cut(s["cal"], bins, right=False)).agg(n=("cal", "size"),
            predicted=("cal", "mean"), observed=(TARGET, "mean")).dropna()
    for k, r in g.iterrows():
        rel.append({"year": y, "bin": str(k), "n": int(r.n), "predicted": round(r.predicted, 4), "observed": round(r.observed, 4)})
out["reliability"] = rel

out["missing"] = [{"feature": f, **{pp: round(float(df.loc[df.period == pp, f].isna().mean()), 4)
                   for pp in ["2022-24", "2025", "2026"]}} for f in feats]

def psi(a, b, k=10):
    a, b = a.dropna(), b.dropna()
    if len(a) < 50 or len(b) < 50 or a.nunique() < 2: return None
    e = np.unique(np.quantile(a, np.linspace(0, 1, k + 1))); e[0], e[-1] = -np.inf, np.inf
    pa = np.histogram(a, e)[0] / len(a) + 1e-4; pb = np.histogram(b, e)[0] / len(b) + 1e-4
    return float(np.sum((pb - pa) * np.log(pb / pa)))
ref = df[df.period == "2022-24"]
out["drift"] = [{"feature": f, "psi_2025": psi(ref[f], df[df.period == "2025"][f]),
                 "psi_2026": psi(ref[f], df[df.period == "2026"][f])} for f in feats]
out["score_psi"] = {"2025": psi(ref["raw"], df[df.period == "2025"]["raw"]),
                    "2026": psi(ref["raw"], df[df.period == "2026"]["raw"])}
json.dump(out, open("monitoring_summary.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
