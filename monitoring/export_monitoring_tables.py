# Wildfire project - export SMALL summary tables for the monitoring dashboard.
# Run in Colab after mounting Drive. Writes monitoring_summary.json (no raw rows, safe to share).
import os, json, warnings, joblib
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive")
CSV    = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
VARIANT = os.environ.get("WF_VARIANT", "nosat")   # "nosat" = main model (20 features); "full" = 24-feature satellite model
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
if VARIANT == "full":
    info = json.load(open(find(INFO))); cal = json.load(open(find(CAL)))
    feats, thr = info["features"], float(info["threshold_recall"])
    model = joblib.load(find(MODEL))
else:
    # main model: no satellite columns. Needs its own calibrator and threshold, so both are fitted here the same way as before:
    # model trained to 2021 -> scores on 2022-24 -> Platt curve and a threshold for ~65% recall; reporting model trained to 2024.
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegression
    SAT = ["modis_count_early7d", "modis_max_frp_early7d", "viirs_count_early7d", "viirs_max_frp_early7d"]
    feats = [f for f in json.load(open(find(INFO)))["features"] if f not in SAT]
    P = dict(n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=31, is_unbalance=True, random_state=42, verbose=-1)
    ok = df["LATITUDE"].between(41, 84) & df["LONGITUDE"].between(-142, -52) & df[TARGET].notna()
    df = df[ok].reset_index(drop=True); yy = df[TARGET].astype(int).values; yr = df[YEAR].values
    m21 = LGBMClassifier(**P).fit(df.loc[yr <= 2021, feats], yy[yr <= 2021])
    va = (yr >= 2022) & (yr <= 2024); sv = m21.predict_proba(df.loc[va, feats])[:, 1]
    thr = float(np.sort(sv[yy[va] == 1])[int(0.35 * (yy[va] == 1).sum())])
    pv = np.clip(sv, EPS, 1 - EPS); lr = LogisticRegression(C=1e6).fit(np.log(pv / (1 - pv)).reshape(-1, 1), yy[va])
    cal = {"coef": float(lr.coef_[0][0]), "intercept": float(lr.intercept_[0])}
    model = LGBMClassifier(**P).fit(df.loc[yr <= 2024, feats], yy[yr <= 2024])
    print(f"no-satellite model: threshold {thr:.3f}, calibrator {cal}")

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

out = {"variant": VARIANT, "features": len(feats), "threshold": thr, "note": "2022-24 scores come from a model trained through 2024 (in-sample); 2025+ is forward."}
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
dest = os.path.join(FOLDER, f"monitoring_summary_v4_{VARIANT}.json")
json.dump(out, open(dest, "w"), indent=1, default=float)
print("Saved full summary to:", dest)
print("Score drift (PSI):", out["score_psi"], "| features over 0.10:", [d["feature"] for d in out["drift"] if max(d["psi_2025"] or 0, d["psi_2026"] or 0) > 0.10])
