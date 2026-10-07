# 1) Road distance x people grid: where exactly does the frozen model (train<=2021, alert line 0.695, test 2022-26) miss big fires, and are its scores too high or too low there?
# 2) Segment-specific calibration: keep the same model, but fit a separate score->chance curve (Platt) per segment on a calibration period and score the later test years.
#    Pooled PR-AUC can change because fires from different segments are re-ordered against each other. Kept only if it gains on BOTH splits with a range above zero.
#    Split A: train<=2018, calibrate on 2019-21, test 2022-24.   Split B: train<=2021, calibrate on 2022-24, test 2025+.
import os, json, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
warnings.filterwarnings("ignore")
FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive"); CSV = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
LAT, LON, TARGET = "LATITUDE", "LONGITUDE", "is_big_fire"; THR = 0.695; NB = 300; EPS = 1e-4; MIN_FIRES, MIN_BIG = 300, 30
P = dict(n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=31, is_unbalance=True, random_state=42, verbose=-1)
def find(name, must=True):
    for root, _, files in os.walk(FOLDER):
        if name in files: return os.path.join(root, name)
    if must: raise FileNotFoundError(name)
feats = json.load(open(find("final_model_v14.6_nosat_info.json")))["features"] + [LAT, LON]
df = pd.read_csv(find(CSV), low_memory=False)
df = df[df[LAT].between(41, 84) & df[LON].between(-142, -52)].dropna(subset=[TARGET]); df["_row"] = df.index; df = df.reset_index(drop=True)
fp = find("province_filled_v4.csv", must=False)
if fp: df = df.merge(pd.read_csv(fp).rename(columns={"row": "_row"}), on="_row", how="left")
y = df[TARGET].astype(int).values; yr = df["year"].values; road, pop = df["dist_to_road_m"].values, df["pop_within_25km"].values
rb = np.select([road < 1000, road < 5000, road < 20000], ["road<1km", "road1-5km", "road5-20km"], "road>20km")
pb = np.select([pop == 0, pop <= 100], ["pop0", "pop1-100"], "pop>100")
cause = df["CAUSE"].astype(str).str.strip().str[0] if "CAUSE" in df.columns else pd.Series([""] * len(df))
SCHEMES = {"road band (4)": rb, "road x people (12)": np.char.add(np.char.add(rb.astype(str), "|"), pb.astype(str))}
if "province_filled" in df.columns: SCHEMES["province"] = df["province_filled"].fillna("unknown").astype(str).values
SCHEMES["cause (needs cause known)"] = np.where(cause.isin(["H", "N"]), cause, "other").astype(str)
logit = lambda s: np.log(np.clip(s, EPS, 1 - EPS) / (1 - np.clip(s, EPS, 1 - EPS)))
def platt(s, yy): lr = LogisticRegression(C=1e6).fit(logit(s).reshape(-1, 1), yy); return float(lr.coef_[0][0]), float(lr.intercept_[0])
def apply(c, s): return 1 / (1 + np.exp(-(c[0] * logit(s) + c[1])))
pd.set_option("display.width", 230); pd.set_option("display.max_rows", 200)

# ---------- 1) grid
trm, tem = yr <= 2021, yr >= 2022
s = LGBMClassifier(**P).fit(df.loc[trm, feats], y[trm]).predict_proba(df[feats])[:, 1]
yt, st = y[tem], s[tem]; a = st >= THR; miss = (yt == 1) & ~a; lab = SCHEMES["road x people (12)"][tem]; rows = []
for k in sorted(set(lab)):
    mk = lab == k
    if mk.sum() < 200 or yt[mk].sum() < 20: continue
    yy, ss, aa = yt[mk], st[mk], a[mk]
    rows.append((k, mk.sum(), 100 * mk.mean(), 100 * yy.mean(), 100 * ss.mean(), roc_auc_score(yy, ss), average_precision_score(yy, ss), 100 * (aa & (yy == 1)).sum() / yy.sum(), 100 * (aa & (yy == 1)).sum() / max(aa.sum(), 1), 100 * miss[mk].sum() / miss.sum()))
G = pd.DataFrame(rows, columns=["cell", "fires", "% of fires", "% big", "avg raw score x100", "ROC-AUC", "PR-AUC", "recall %", "precision %", "% of ALL misses"])
G[["ROC-AUC", "PR-AUC"]] = G[["ROC-AUC", "PR-AUC"]].round(3)
print(f"1) ROAD x PEOPLE GRID (train<=2021, test 2022-26: {tem.sum():,} fires, {yt.sum():,} big, {miss.sum():,} missed, alert line {THR})")
print(G.sort_values("% of ALL misses", ascending=False).round({"% of fires": 1, "% big": 1, "avg raw score x100": 1, "recall %": 1, "precision %": 1, "% of ALL misses": 1}).to_string(index=False))
print("   (raw score is not a probability; compare the 'avg raw score' with '% big' only after the Platt step in part 2)")

# ---------- 2) segment-specific calibration
rng = np.random.default_rng(0)
def run(name, trm, calm, tem):
    s = LGBMClassifier(**P).fit(df.loc[trm, feats], y[trm]).predict_proba(df[feats])[:, 1]; g = platt(s[calm], y[calm]); yt, st = y[tem], s[tem]
    base_pr, base_roc = average_precision_score(yt, st), roc_auc_score(yt, st); gp = apply(g, st)
    print(f"\n   {name}: {tem.sum():,} test fires, {yt.sum():,} big | raw score PR {base_pr:.3f} ROC {base_roc:.3f} | global Platt Brier {brier_score_loss(yt, gp):.4f} (same ranking)")
    out = []
    for sname, labs in SCHEMES.items():
        cal = {}
        for k in set(labs[calm]):
            m = calm & (labs == k)
            if m.sum() >= MIN_FIRES and y[m].sum() >= MIN_BIG: cal[k] = platt(s[m], y[m])
        p = np.array([apply(cal.get(k, g), v) for k, v in zip(labs[tem], st)])
        d = []
        for _ in range(NB):
            b = rng.integers(0, len(yt), len(yt))
            if yt[b].sum(): d.append(average_precision_score(yt[b], p[b]) - average_precision_score(yt[b], st[b]))
        o = np.argsort(-p)[:int(.15 * len(p))]; o0 = np.argsort(-st)[:int(.15 * len(st))]
        out.append((sname, len(cal), f"{average_precision_score(yt, p):.3f}", f"{roc_auc_score(yt, p):.3f}", f"{np.mean(d):+.3f} ({np.percentile(d, 2.5):+.3f} to {np.percentile(d, 97.5):+.3f})",
                    f"{yt[o].sum() / yt.sum():.0%} vs {yt[o0].sum() / yt.sum():.0%}", f"{brier_score_loss(yt, p):.4f}", np.percentile(d, 2.5), np.percentile(d, 97.5)))
    return out
print("\n2) SEGMENT-SPECIFIC CALIBRATION (each segment gets its own score->chance curve; segments too small use the global curve)")
A = run("Split A: train<=2018, calibrate 2019-21", yr <= 2018, (yr >= 2019) & (yr <= 2021), (yr >= 2022) & (yr <= 2024))
B = run("Split B: train<=2021, calibrate 2022-24", yr <= 2021, (yr >= 2022) & (yr <= 2024), yr >= 2025)
rows = []
for x, z in zip(A, B):
    rows.append((x[0], x[1], x[2], x[3], x[4], x[5], x[6], z[2], z[3], z[4], z[5], z[6], "GAINS" if x[7] > 0 and z[7] > 0 else ""))
print(pd.DataFrame(rows, columns=["segmentation", "curves A", "PR A", "ROC A", "PR gain A", "top15% A (seg vs raw)", "Brier A", "PR B", "ROC B", "PR gain B", "top15% B (seg vs raw)", "Brier B", "verdict"]).to_string(index=False))
print("\nDone. Paste all of this output back.")
