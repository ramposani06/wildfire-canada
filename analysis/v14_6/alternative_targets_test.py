# Is the model good at other, more dispatcher-like questions than ">100 ha"?   Same 22 features (v14.7), same split protocol.
#  Size targets:      final size over 4 / 10 / 100 ha  ("got past the first attack" is closer to 4-10 ha than 100 ha)
#  Duration targets:  days from report to OUT_DATE over 3 / 7 / 14 days. OUT_DATE is used as a LABEL only, never as an input.
# Duration needs a filled-in out date, so coverage is shown first and only fires with a valid date are used (2026 is left out: many fires are still burning).
import os, json, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive")
CSV  = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
INFO = "final_model_v14.6_nosat_info.json"
LAT, LON, TARGET = "LATITUDE", "LONGITUDE", "is_big_fire"
P = dict(n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=31, is_unbalance=True, random_state=42, verbose=-1)
def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files: return os.path.join(root, name)
    raise FileNotFoundError(name)
feats = json.load(open(find(INFO)))["features"] + [LAT, LON]
df = pd.read_csv(find(CSV), low_memory=False)
df = df[df[LAT].between(41, 84) & df[LON].between(-142, -52)].dropna(subset=[TARGET]).reset_index(drop=True)
yr = df["year"].values
size = next(c for c in df.columns if c.upper() == "SIZE_HA")
rep = pd.to_datetime(df["REP_DATE"].astype(str).str[:10], errors="coerce"); out = pd.to_datetime(df["OUT_DATE"].astype(str).str[:10], errors="coerce")
df["dur"] = (out - rep).dt.days
ok = df["dur"].notna() & (df["dur"] >= 0) & (df["dur"] < 400)
print("Coverage of a usable out date (share of fires, by year):")
cov = pd.Series(ok.values).groupby(yr).mean().mul(100).round(0).astype(int)
print(cov.loc[2004:].to_string().replace("\n", " | "))
print("Median days to out among valid, by year (2019-2025):", df[ok].groupby("year")["dur"].median().loc[2019:2025].to_dict())

def lift_row(label, y, tr, te, valid=None):
    if valid is not None: tr, te = tr & valid, te & valid
    yt = y[te]
    if yt.sum() < 30: print(f"  {label}: too few positives"); return
    s = LGBMClassifier(**P).fit(df.loc[tr, feats], y[tr]).predict_proba(df.loc[te, feats])[:, 1]
    base = yt.mean(); pr = average_precision_score(yt, s); o = np.argsort(-s); top = yt[o[:int(.15 * len(s))]].sum() / yt.sum()
    r = np.random.RandomState(1); b = []
    for _ in range(200):
        i = r.randint(0, len(yt), len(yt))
        if yt[i].sum(): b.append(average_precision_score(yt[i], s[i]))
    print(f"  {label:<22} n={te.sum():>6,} base {base:5.1%} | ROC {roc_auc_score(yt, s):.3f} | PR {pr:.3f} ({np.percentile(b,2.5):.3f}-{np.percentile(b,97.5):.3f}) | lift {pr/base:4.1f}x | top 15% catches {top:.0%}", flush=True)

SPL = [("train<=2021 -> test 2022-24", yr <= 2021, (yr >= 2022) & (yr <= 2024)), ("train<=2024 -> test 2025+ ", yr <= 2024, yr >= 2025)]
print("\n=== SIZE targets ===")
for nm, tr, te in SPL:
    print(nm)
    for ha in (4, 10, 100): lift_row(f"over {ha} ha", (df[size] > ha).astype(int).values, tr, te)
print("\n=== DURATION targets (fires with a valid out date; 2026 excluded) ===")
valid = ok.values & (yr <= 2025)
for nm, tr, te in SPL:
    print(nm)
    for dd in (3, 7, 14): lift_row(f"burning over {dd} days", (df["dur"] > dd).astype(int).values, tr, te, valid)
print("\nDone. Paste all of this output back.")
