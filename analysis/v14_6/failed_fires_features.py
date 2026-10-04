# Use the FAILED fires as a dataset of their own: among fires the model got wrong (missed big fires vs false alarms), which columns tell them apart?
# Positive = MISSED big fire, negative = false alarm (small fire the model flagged). Year-by-year held-out folds. Everything is out-of-sample (2022-26).
# Columns that describe the fire AFTER it happened (size, area, end dates, duration, target) are excluded. Satellite counts are marked (end of report day).
import os, re, json, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
warnings.filterwarnings("ignore")
FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive"); CSV = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
LAT, LON, TARGET = "LATITUDE", "LONGITUDE", "is_big_fire"
P = dict(n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=31, is_unbalance=True, random_state=42, verbose=-1)
def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files: return os.path.join(root, name)
    raise FileNotFoundError(name)
feats = json.load(open(find("final_model_v14.6_nosat_info.json")))["features"] + [LAT, LON]
df = pd.read_csv(find(CSV), low_memory=False)
df = df[df[LAT].between(41, 84) & df[LON].between(-142, -52)].dropna(subset=[TARGET]).reset_index(drop=True)
y = df[TARGET].astype(int).values; yr = df["year"].values; tr, va = yr <= 2021, (yr >= 2022) & (yr <= 2024)
m = LGBMClassifier(**P).fit(df.loc[tr, feats], y[tr]); df["score"] = m.predict_proba(df[feats])[:, 1]
thr = float(np.sort(df.loc[va & (y == 1), "score"].values)[int(0.35 * (va & (y == 1)).sum())])
d = df[yr >= 2022].copy(); d["alert"] = d["score"] >= thr
d["rep_dayofyear"] = pd.to_datetime(d["REP_DATE"].astype(str).str[:10], errors="coerce").dt.dayofyear
if "CAUSE" in d.columns:
    c0 = d["CAUSE"].astype(str).str.strip().str[0]; d["cause_human"] = (c0 == "H").astype(int); d["cause_natural"] = (c0 == "N").astype(int)
F = d[(d.alert) & (d[TARGET] == 0) | (~d.alert) & (d[TARGET] == 1)].copy(); F["missed"] = F[TARGET].astype(int)   # missed big=1; false alarm=0
print(f"alert line {thr:.3f} | failed fires {len(F):,}: missed big {int(F.missed.sum()):,} vs false alarms {int((1-F.missed).sum()):,}")
bad = re.compile(r"size|_ha\b|area|out_|outdate|end|duration|big|target|score|alert|group|missed|orig|^year$|id$|^index|unnamed|lat_|lon_|rep_date|date$", re.I)
cand = [c for c in F.columns if pd.api.types.is_numeric_dtype(F[c]) and F[c].nunique() > 2 and F[c].notna().mean() > 0.5 and (c in feats or not bad.search(c))]
cand = [c for c in cand if c not in ("missed", TARGET, "alert", "score")]
sat = [c for c in cand if "early7d" in c]; non_in = [c for c in cand if c not in feats]
print(f"{len(cand)} candidate columns: {len([c for c in cand if c in feats])} model inputs + {len(non_in)} others. Dropped as 'after the fact': ", sorted(c for c in F.columns if pd.api.types.is_numeric_dtype(F[c]) and bad.search(c) and c not in feats)[:25])
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
Fy = F["year"].values; yy = F["missed"].values
def cv(cols, reps=1, imp=False):
    oof = np.full(len(F), np.nan); pim = pd.Series(0.0, index=cols); gain = pd.Series(0.0, index=cols); k = 0
    for y0 in sorted(set(Fy)):
        te, trn = Fy == y0, Fy != y0
        if te.sum() < 50 or yy[te].sum() < 10: continue
        mm = LGBMClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, random_state=1, verbose=-1).fit(F.loc[trn, cols], yy[trn])
        oof[te] = mm.predict_proba(F.loc[te, cols])[:, 1]
        if imp:
            r = permutation_importance(mm, F.loc[te, cols], yy[te], scoring="roc_auc", n_repeats=3, random_state=0); pim += pd.Series(r.importances_mean, index=cols); gain += pd.Series(mm.booster_.feature_importance("gain"), index=cols) / mm.booster_.feature_importance("gain").sum(); k += 1
    ok = ~np.isnan(oof); return roc_auc_score(yy[ok], oof[ok]), (pim / max(k, 1), gain / max(k, 1))
print("\n1) How well can each column set separate MISSED big from FALSE ALARM (0.5 = no better than a coin; held-out years):")
yrs = {y0: roc_auc_score(yy[Fy == y0], F["score"].values[Fy == y0]) for y0 in sorted(set(Fy)) if yy[Fy == y0].sum() >= 10 and (1 - yy[Fy == y0]).sum() >= 10}
print(f"   model's own score (missed big tend to score LOWER, so lower = missed): AUC of -score = {np.mean([1 - a for a in yrs.values()]):.3f}")
a_in, _ = cv([c for c in cand if c in feats]); a_non, _ = cv(non_in); a_all, (pim, gain) = cv(cand, imp=True)
print(f"   the 22 model inputs only:           AUC {a_in:.3f}\n   all OTHER columns (not model inputs): AUC {a_non:.3f}\n   everything together:                 AUC {a_all:.3f}")
if sat: a_ns, _ = cv([c for c in cand if c not in sat]); print(f"   everything except satellite counts:  AUC {a_ns:.3f}   (satellite adds {a_all - a_ns:+.3f})")

print("\n2) Most important columns (permutation importance = AUC lost when the column is shuffled; held-out years):")
T = pd.DataFrame({"perm importance": pim, "gain share": gain}).assign(kind=lambda t: ["model input" if c in feats else ("satellite (end of report day)" if c in sat else "not an input") for c in t.index]).sort_values("perm importance", ascending=False)
print(T.head(25).round(4).to_string())

print("\n3) Single-column AUC, missed big vs false alarm (above 0.5: bigger value = more likely a MISSED big fire):")
rows = []
for c in cand:
    v = F[c]; ok = v.notna().values
    if ok.sum() > 100 and v[ok].nunique() > 2: a = roc_auc_score(yy[ok], v[ok]); rows.append((c, "model input" if c in feats else ("satellite" if c in sat else "not an input"), a, abs(a - 0.5)))
S = pd.DataFrame(rows, columns=["column", "kind", "AUC alone", "abs"]).sort_values("abs", ascending=False).drop(columns="abs"); print(S.head(20).round(3).to_string(index=False))
T.to_csv(os.path.join(FOLDER, "failed_fires_feature_importance.csv")); print("\nSaved failed_fires_feature_importance.csv to Drive.\nDone. Paste all of this output back.")
