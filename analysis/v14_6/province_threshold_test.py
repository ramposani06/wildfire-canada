# One national alert line vs a separate line per province (and per province + near-road tier).
# Honest setup: ONE frozen model (v14.7 trained to 2021). Lines are fitted on its 2022-24 scores (65% recall inside each group),
# then applied to 2025+ fires the model never saw. Groups with fewer than 40 big fires in 2022-24 use the national line.
# Fair comparison: the national line is also moved to flag the SAME share of fires as the per-province rule, so we see if it is really better, not just flagging more.
import os, json, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
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
fill = pd.read_csv(find("province_filled_v4.csv"))
df = raw[raw[LAT].between(41, 84) & raw[LON].between(-142, -52)].dropna(subset=[TARGET]).merge(fill, on="row", how="left").reset_index(drop=True)
y = df[TARGET].astype(int).values; yr = df["year"].values; prov = df["province_filled"].fillna("?").values
near = (df["dist_to_road_m"] <= 5000).values
m = LGBMClassifier(**P).fit(df.loc[yr <= 2021, feats], y[yr <= 2021]); s = m.predict_proba(df[feats])[:, 1]
va, te = (yr >= 2022) & (yr <= 2024), yr >= 2025
def thr65(sc, yy): return float(np.sort(sc[yy == 1])[int(0.35 * (yy == 1).sum())])
t_nat = thr65(s[va], y[va]); MINBIG = 40
def build(group):
    out = {}
    for g in np.unique(group):
        mk = va & (group == g)
        if y[mk].sum() >= MINBIG: out[g] = thr65(s[mk], y[mk])
    return out
def apply(group, thr): return s >= np.array([thr.get(g, t_nat) for g in group])
tiers = {"province": prov, "province + near-road tier": np.array([f"{p}|{'near' if n else 'far'}" for p, n in zip(prov, near)])}
yt = y[te]
def summary(name, fl):
    f = fl[te]; tp = (f & (yt == 1)).sum()
    return f"{name:<44} flags {f.mean():5.1%} | catches {tp/yt.sum():5.1%} of big fires | precision {tp/max(f.sum(),1):5.1%}"
print(f"model trained to 2021 | national line {t_nat:.3f} (65% recall on 2022-24) | test 2025+: {te.sum():,} fires, {yt.sum():,} big")
fl_nat = s >= t_nat; print("\n" + summary("ONE national line", fl_nat))
for name, grp in tiers.items():
    thr = build(grp); fl = apply(grp, thr); print(summary(f"separate line per {name} ({len(thr)} groups)", fl))
    # same alert budget with a single national line
    q = float(np.quantile(s[te], 1 - fl[te].mean())); fq = s >= q
    print(summary(f"   national line moved to flag the same share ({q:.2f})", fq))
    if name == "province":
        rows = []
        for g in sorted(set(prov[te]), key=lambda k: -y[te][prov[te] == k].sum()):
            mk = te & (prov == g); b = y[mk].sum()
            if b < 25: continue
            rows.append((g, int(b), thr.get(g, t_nat), (fl_nat[mk] & (y[mk] == 1)).sum() / b, fl_nat[mk].mean(), (fl[mk] & (y[mk] == 1)).sum() / b, fl[mk].mean(), (fl[mk] & (y[mk] == 1)).sum() / max(fl[mk].sum(), 1)))
        T = pd.DataFrame(rows, columns=["province", "big", "own line", "catches (national)", "alerts (national)", "catches (own)", "alerts (own)", "precision (own)"]).set_index("province")
        print("\nBy province, 2025+  (target recall was 65%):"); print(T.round(2).to_string()); print()
print("\nDone. Paste all of this output back.")
