# Fill the missing province names (resolved_province is empty for ~7,500 fires in 2025+), check the fill, then redo the province tables.
# Fill order: 1) province_encoded -> province (if that code always means one province in the known rows)  2) nearest known fires by latitude/longitude.
# The filled column is saved to Drive as province_filled_v4.csv (row number + province) so the data can be fixed properly at the next rebuild.
import os, json, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
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
raw = pd.read_csv(find(CSV), low_memory=False); raw["row"] = raw.index
df = raw[raw[LAT].between(41, 84) & raw[LON].between(-142, -52)].dropna(subset=[TARGET]).reset_index(drop=True)
known = df["resolved_province"].notna(); print(f"province known for {known.sum():,} of {len(df):,} fires; empty by year:", df[~known].groupby("year").size().to_dict())

# 1) code -> province
ct = pd.crosstab(df.loc[known, "province_encoded"], df.loc[known, "resolved_province"])
pur = (ct.max(axis=1) / ct.sum(axis=1)); code2prov = ct.idxmax(axis=1)
print(f"\nprovince_encoded codes: {len(ct)} | codes that always mean ONE province (purity>=99%): {(pur>=0.99).sum()} | lowest purity {pur.min():.2f}")
print("Mapping:", {int(k): v for k, v in code2prov.items()})
# 2) nearest known fires by lat/lon (scaled so a degree of longitude is shorter at high latitude)
def xy(d): return np.c_[d[LAT].values, d[LON].values * np.cos(np.radians(d[LAT].values))]
knn = KNeighborsClassifier(n_neighbors=7, weights="distance").fit(xy(df[known]), df.loc[known, "resolved_province"])
acc = cross_val_score(KNeighborsClassifier(n_neighbors=7, weights="distance"), xy(df[known]), df.loc[known, "resolved_province"], cv=5).mean()
print(f"nearest-fire fill, 5-fold accuracy on known fires: {acc:.1%}")
u = df[~known]; by_code = u["province_encoded"].map(code2prov); by_knn = pd.Series(knn.predict(xy(u)), index=u.index)
ok = by_code.notna() & (u["province_encoded"].map(pur) >= 0.99)
print(f"empty fires: {len(u):,} | fillable by code: {ok.sum():,} | where code and location agree: {(by_code[ok] == by_knn[ok]).mean():.1%}")
df["prov"] = df["resolved_province"]; df.loc[u.index, "prov"] = np.where(ok, by_code, by_knn)
df[["row", "prov"]].rename(columns={"prov": "province_filled"}).to_csv(os.path.join(FOLDER, "province_filled_v4.csv"), index=False)
print("Saved province_filled_v4.csv to Drive. Province counts now:", df["prov"].value_counts().head(13).to_dict())

# redo tables
y = df[TARGET].astype(int).values; yr = df["year"].values; prov = df["prov"].values
def pr(a, s): return average_precision_score(a, s) if 0 < a.sum() < len(a) else np.nan
def roc(a, s): return roc_auc_score(a, s) if 0 < a.sum() < len(a) else np.nan
pd.set_option("display.width", 220)
m21 = LGBMClassifier(**P).fit(df.loc[yr <= 2021, feats], y[yr <= 2021]); s21 = m21.predict_proba(df[feats])[:, 1]
va = (yr >= 2022) & (yr <= 2024); thr = float(np.sort(s21[va & (y == 1)])[int(0.35 * (va & (y == 1)).sum())])
print(f"\n=== Province table, frozen model (train<=2021), fires 2022-26, alert line {thr:.3f} (every province now filled) ===")
rows = []
for pv in pd.Series(prov[yr >= 2022]).value_counts().index:
    mk = (prov == pv) & (yr >= 2022)
    if mk.sum() < 200 or y[mk].sum() < 20: continue
    fl = s21[mk] >= thr; bg = y[mk] == 1
    rows.append((pv, mk.sum(), int(y[mk].sum()), y[mk].mean(), roc(y[mk], s21[mk]), pr(y[mk], s21[mk]), pr(y[mk], s21[mk]) / y[mk].mean(), (fl & bg).sum() / bg.sum(), (fl & bg).sum() / max(fl.sum(), 1), fl.mean()))
T = pd.DataFrame(rows, columns=["province", "fires", "big", "base rate", "ROC", "PR", "lift", "catches %", "precision %", "alerts %"]).set_index("province")
for c in ("catches %", "precision %", "alerts %"): T[c] = (T[c] * 100).round(0)
print(T.sort_values("big", ascending=False).round(3).to_string())
print(f"\n=== Province x year PR-AUC (big fires in brackets; '-' = under 15 big fires) ===")
provs = list(T.sort_values("big", ascending=False).index[:10]); grid = []
for pv in provs:
    r = {"province": pv}
    for Y in (2022, 2023, 2024, 2025, 2026):
        mk = (prov == pv) & (yr == Y); nb = int(y[mk].sum()); r[Y] = f"{pr(y[mk], s21[mk]):.2f} ({nb})" if nb >= 15 and nb < mk.sum() else "-"
    grid.append(r)
print(pd.DataFrame(grid).set_index("province").to_string())
print("\n=== Final model (train<=2024) on 2025+, by province ===")
te = yr >= 2025; s24 = LGBMClassifier(**P).fit(df.loc[yr <= 2024, feats], y[yr <= 2024]).predict_proba(df.loc[te, feats])[:, 1]; rows = []
for pv in pd.Series(prov[te]).value_counts().index:
    mk = prov[te] == pv; yy = y[te][mk]
    if mk.sum() < 150 or yy.sum() < 10: continue
    rows.append((pv, mk.sum(), int(yy.sum()), yy.mean(), roc(yy, s24[mk]), pr(yy, s24[mk]), pr(yy, s24[mk]) / yy.mean()))
print(pd.DataFrame(rows, columns=["province", "fires", "big", "base rate", "ROC", "PR", "lift"]).set_index("province").round(3).sort_values("big", ascending=False).to_string())
print("\nDone. Paste all of this output back.")
