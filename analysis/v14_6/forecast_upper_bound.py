# Forecast-weather UPPER-BOUND test (run in Colab after mounting Drive).
# Question: if we knew the weather of the report day and the next 2 days, how much would PR-AUC rise?
# Uses ACTUAL weather (Open-Meteo archive), i.e. a perfect forecast. Real forecasts will do worse.
# If even this upper bound adds little, real forecasts are not worth building.
# Design: same fires, same model, with vs without the new features (no-satellite model, 20 features).
# Train = random sample of 2010-2024 fires; test = all 2025+ fires. Resumable: rerun after a rate limit.
import os, json, time, warnings
import numpy as np, pandas as pd, requests
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")

FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive")
CSV    = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
INFO   = "final_model_v14.6_nosat_info.json"
OUT    = os.path.join(FOLDER, "forecast_upper_bound_weather.csv")
MAX_BATCH, PACE, BACKOFF, MAX_SECONDS = 30, 0.3, 60, int(os.environ.get("WF_MAX_SECONDS", 3000))
VARS = ["temperature_2m_max", "precipitation_sum", "wind_speed_10m_max", "wind_gusts_10m_max", "relative_humidity_2m_mean"]
NEW = ["next3_temp_max", "next3_wind_max", "next3_gust_max", "next3_precip_sum", "next3_rh_min", "next3_wind_mean"]
LAT, LON, TARGET = "LATITUDE", "LONGITUDE", "is_big_fire"

def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files: return os.path.join(root, name)
    raise FileNotFoundError(name)

df = pd.read_csv(find(CSV), low_memory=False)
df = df[df[LAT].between(41, 84) & df[LON].between(-142, -52)].dropna(subset=[TARGET]).copy()
df["_d"] = pd.to_datetime(df["REP_DATE"].astype(str).str[:10], errors="coerce")
df = df[df["_d"].between("2010-01-01", "2026-09-20")].copy()
df["_id"] = df.index.astype(str)
N_TEST = int(os.environ.get("WF_N_TEST", 4000))
N_TRAIN = int(os.environ.get("WF_N_TRAIN", 6000))
done = set(pd.read_csv(OUT)["_id"].astype(str)) if os.path.exists(OUT) else set()
def pick(pool, n):   # fires already fetched first, then a random top-up
    have = pool[pool["_id"].isin(done)]
    rest = pool[~pool["_id"].isin(done)]
    take = have.sample(min(n, len(have)), random_state=7)
    if len(take) < n: take = pd.concat([take, rest.sample(min(n - len(take), len(rest)), random_state=7)])
    return take
test = pick(df[df["year"] >= 2025], N_TEST)
train = pick(df[df["year"] <= 2024], N_TRAIN)
use = pd.concat([train, test]); print(f"fires to use: train {len(train):,} + test {len(test):,}")
todo = use[~use["_id"].isin(done)]
print(f"already fetched {len(done):,}; remaining {len(todo):,}")
t0 = time.time(); cols = ["_id"] + NEW

def fetch(g, d):
    p = {"latitude": ",".join(g[LAT].astype(str)), "longitude": ",".join(g[LON].astype(str)),
         "start_date": d.strftime("%Y-%m-%d"), "end_date": (d + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
         "daily": VARS, "timezone": "auto"}
    for a in range(5):
        try:
            r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=p, timeout=45)
        except requests.exceptions.RequestException:
            time.sleep(5 * (a + 1)); continue          # network hiccup: wait and retry
        if r.status_code == 429: time.sleep(BACKOFF * (a + 1)); continue
        if r.status_code != 200: return None
        res = r.json(); res = res if isinstance(res, list) else [res]
        if len(res) != len(g): return None
        rows = []
        for fid, x in zip(g["_id"], res):
            q = x.get("daily", {}); f = lambda k: np.array(q.get(k, [np.nan]), dtype=float)
            rows.append([fid, np.nanmax(f("temperature_2m_max")), np.nanmax(f("wind_speed_10m_max")),
                         np.nanmax(f("wind_gusts_10m_max")), np.nansum(f("precipitation_sum")),
                         np.nanmin(f("relative_humidity_2m_mean")), np.nanmean(f("wind_speed_10m_max"))])
        return rows
    return "rate"

from concurrent.futures import ThreadPoolExecutor
jobs = [(grp.iloc[i:i + MAX_BATCH], d) for d, grp in todo.groupby("_d") for i in range(0, len(grp), MAX_BATCH)]
n_done = 0; n_todo = len(todo); n_fail = 0; n_batch = 0
with ThreadPoolExecutor(max_workers=4) as ex:
    for k in range(0, len(jobs), 40):
        if time.time() - t0 > MAX_SECONDS: print("time limit for this run reached; rerun to continue"); break
        results = list(ex.map(lambda j: fetch(*j), jobs[k:k + 40]))
        if any(r == "rate" for r in results): print("rate limited; stopping. Rerun in a few minutes."); break
        for r in results:
            if r:
                pd.DataFrame(r, columns=cols).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
                n_done += len(r)
            else: n_fail += 1
        el = time.time() - t0; rate = n_done / max(el, 1); eta = (n_todo - n_done) / max(rate, 1e-9) / 60
        print(f"progress: {n_done:,}/{n_todo:,} fires ({n_done / max(n_todo, 1):.0%}) | {rate:.1f} fires/s | about {eta:.0f} min left | failed batches {n_fail}", flush=True)
got = pd.read_csv(OUT).drop_duplicates("_id"); got["_id"] = got["_id"].astype(str)
print(f"fetched so far: {len(got):,} of {len(use):,}")
if len(got) < 0.97 * len(use):
    raise SystemExit("Not finished. Run this cell again (it resumes). Analysis runs when 97% is fetched.")

# ---- A/B ----
feats = json.load(open(find(INFO)))["features"]
m = use.merge(got, on="_id", how="inner").dropna(subset=NEW)
tr, te = m[m["year"] <= 2024], m[m["year"] >= 2025]
PARAMS = dict(n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=31, is_unbalance=True, random_state=42, verbose=-1)
ytr, yte = tr[TARGET].astype(int).values, te[TARGET].astype(int).values
print(f"\ntrain {len(tr):,} | test {len(te):,} ({yte.mean():.3f} big)")
sA = LGBMClassifier(**PARAMS).fit(tr[feats], ytr).predict_proba(te[feats])[:, 1]
sB = LGBMClassifier(**PARAMS).fit(tr[feats + NEW], ytr).predict_proba(te[feats + NEW])[:, 1]
r = np.random.RandomState(1); dA, dP = [], []
for _ in range(500):
    i = r.randint(0, len(yte), len(yte))
    if yte[i].sum() == 0: continue
    dA.append(roc_auc_score(yte[i], sB[i]) - roc_auc_score(yte[i], sA[i]))
    dP.append(average_precision_score(yte[i], sB[i]) - average_precision_score(yte[i], sA[i]))
for lab, s in (("Without next-3-day weather (20 features)", sA), ("With next-3-day ACTUAL weather (26 features)", sB)):
    order = np.argsort(-s); top = yte[order[:int(.15 * len(s))]].sum() / yte.sum()
    print(f"{lab:<46} ROC-AUC {roc_auc_score(yte, s):.3f} | PR-AUC {average_precision_score(yte, s):.3f} | top 15% catches {top:.0%}")
print(f"Gain in ROC-AUC {np.mean(dA):+.3f} (95% {np.percentile(dA,2.5):+.3f} to {np.percentile(dA,97.5):+.3f})")
print(f"Gain in PR-AUC  {np.mean(dP):+.3f} (95% {np.percentile(dP,2.5):+.3f} to {np.percentile(dP,97.5):+.3f})")
print("\nHow to read: this is the BEST case (perfect forecast). A real forecast will gain less. "
      "If the PR-AUC gain is under about 0.02, stop; if it is large, the next step is real archived forecasts.")
print("Done. Paste all of this output back.")
