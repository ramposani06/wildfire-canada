# Satellite timing audit (run in Colab after mounting Drive).
# Question: the satellite window runs from 7 days before THROUGH the report day, and report
# dates have no clock time. Detections on the report day may have happened after the fire was
# reported. How much does the score depend on those report-day detections?
# Method: rebuild the 4 satellite features from the raw detections two ways,
#   A = days -7..0 (what the model was trained on), B = days -7..-1 (strictly before the report day),
# then score the 2025+ fires with each and compare.
import os, sqlite3, warnings, joblib, json
import numpy as np, pandas as pd
from sklearn.neighbors import BallTree
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")

FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive")
CSV    = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
MODEL  = os.environ.get("WF_MODEL", "final_model_v14.6_clean.pkl")   # trained to 2024
INFO   = "final_model_v14.6_clean_info.json"
DB     = "wildfire_corrected.db"
RADIUS_KM, TEST_FROM, THRESH = 10.0, 2025, 0.770

def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError(name)

df = pd.read_csv(find(CSV), low_memory=False)
df = df[df["LATITUDE"].between(41, 84) & df["LONGITUDE"].between(-142, -52)].dropna(subset=["is_big_fire"])
df["_d"] = pd.to_datetime(df["REP_DATE"].astype(str).str[:10], errors="coerce")
df = df[(df["year"] >= TEST_FROM) & df["_d"].between("2025-01-01", "2026-12-31")].reset_index(drop=True)
print(f"Test fires 2025+: {len(df):,}  (big rate {df['is_big_fire'].mean():.3f})")

con = sqlite3.connect(find(DB))
def load(table):
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
    frp = next((c for c in cols if "frp" in c.lower()), None)
    print(f"\n{table}: columns {cols}")
    if frp is None:
        print("  no FRP column -> FRP features cannot be rebuilt from this table"); 
    q = f"SELECT latitude, longitude, acq_date{', ' + frp + ' AS frp' if frp else ''} FROM {table} WHERE acq_date >= '2024-12-01'"
    d = pd.read_sql(q, con)
    d["acq_date"] = pd.to_datetime(d["acq_date"].astype(str).str[:10], errors="coerce")
    if "frp" not in d: d["frp"] = np.nan
    print("  detections since 2024-12 by year:", d.groupby(d["acq_date"].dt.year).size().to_dict())
    return d.dropna(subset=["acq_date"])

tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
sens = {}
for name, t in (("modis", "modis_detections"), ("viirs", "viirs_detections")):
    if t in tables:
        sens[name] = load(t)
    else:
        print(f"\n{t} not in the database")
if not any((len(v) and (v['acq_date'].dt.year >= TEST_FROM).any()) for v in sens.values()):
    raise SystemExit("\nThe database has no raw detections for 2025+, so the timing check cannot be done from it. "
                     "Paste this output back; the raw 2025-26 detections would need to be re-downloaded from FIRMS.")

fire_rad = np.radians(df[["LATITUDE", "LONGITUDE"]].values)
rep = df["_d"].values.astype("datetime64[D]")
def rebuild(name, lo, hi):
    d = sens.get(name)
    n = np.zeros(len(df)); mx = np.zeros(len(df))
    if d is None or not len(d): return n, mx
    tree = BallTree(np.radians(d[["latitude", "longitude"]].values), metric="haversine")
    dd = d["acq_date"].values.astype("datetime64[D]"); fr = d["frp"].values
    for i, idx in enumerate(tree.query_radius(fire_rad, r=RADIUS_KM / 6371.0)):
        if len(idx) == 0: continue
        k = (rep[i] - dd[idx]).astype(int)          # days before the report date
        m = idx[(k >= lo) & (k <= hi)]
        n[i] = len(m)
        if len(m) and not np.all(np.isnan(fr[m])): mx[i] = np.nanmax(fr[m])
    return n, mx

feat = {}
for tag, lo in (("A", 0), ("B", 1)):
    for s in ("modis", "viirs"):
        feat[(tag, s)] = rebuild(s, lo, 7)

# 1. does my rebuild reproduce the stored features?
print("\n== 1. Rebuild check (window A vs the stored feature) ==")
for s in ("modis", "viirs"):
    st = df[f"{s}_count_early7d"].fillna(0).values
    a = feat[("A", s)][0]
    print(f"{s}: stored count>0 on {(st>0).sum():,} fires, rebuilt on {(a>0).sum():,}; identical counts on {(st==a).mean():.1%} of fires")

# 2. how much of the signal is on the report day only?
print("\n== 2. Report-day detections ==")
for s in ("modis", "viirs"):
    a, b = feat[("A", s)][0], feat[("B", s)][0]
    has = a > 0
    print(f"{s}: fires with any detection {has.sum():,}; of those, only on the report day (none before): "
          f"{((b == 0) & has).sum():,} ({((b == 0) & has).sum() / max(has.sum(), 1):.0%}); "
          f"detections on the report day = {(a.sum() - b.sum()) / max(a.sum(), 1):.0%} of all counted")
anyA = (feat[("A", "modis")][0] + feat[("A", "viirs")][0]) > 0
anyB = (feat[("B", "modis")][0] + feat[("B", "viirs")][0]) > 0
print(f"Any sensor: {anyA.sum():,} fires with a detection; {anyB.sum():,} still have one without the report day")
if anyA.sum():
    print(f"Big-fire rate: with A-detection {df.loc[anyA,'is_big_fire'].mean():.2f}, "
          f"with B-detection {df.loc[anyB,'is_big_fire'].mean():.2f}, "
          f"report-day-only {df.loc[anyA & ~anyB,'is_big_fire'].mean():.2f}")

# 3. score with each version
print("\n== 3. Score the 2025+ fires with each version of the satellite features ==")
obj = joblib.load(find(MODEL)); model = obj.get("model", obj) if isinstance(obj, dict) else obj
feats = json.load(open(find(INFO)))["features"]
y = df["is_big_fire"].astype(int).values
def score(frame): return model.predict_proba(frame[feats])[:, 1]
def report(label, s):
    flag = s >= THRESH
    print(f"{label:<44} ROC-AUC {roc_auc_score(y, s):.3f} | PR-AUC {average_precision_score(y, s):.3f} | "
          f"0.770 rule: flags {flag.mean():.0%}, catches {(flag & (y==1)).sum()/y.sum():.0%}, precision {(flag & (y==1)).sum()/max(flag.sum(),1):.0%}")
s0 = score(df); report("Stored features (what we reported)", s0)
for tag, label in (("A", "Rebuilt, report day included (A)"), ("B", "Rebuilt, report day removed (B)")):
    f = df.copy()
    for s in ("modis", "viirs"):
        f[f"{s}_count_early7d"], f[f"{s}_max_frp_early7d"] = feat[(tag, s)]
    report(label, score(f))
f = df.copy()
for c in ("modis_count_early7d", "modis_max_frp_early7d", "viirs_count_early7d", "viirs_max_frp_early7d"):
    f[c] = 0.0
report("All 4 satellite features set to 0 (floor)", score(f))
print("\nHow to read: if B is close to A, the report-day detections are not carrying the result. "
      "If B drops toward the floor row, the score leans on same-day detections that may come after the report.")
print("Done. Paste all of this output back.")
