# The model misses most big fires that start near roads / people. Test: how good is it INSIDE that segment, and does a better setup help?
# Segments: A = within 5 km of a road;  B = 1,000+ people within 25 km.
# Setups (same 22 features): 1) national model  2) specialist trained only on the segment  3) national model with segment fires counted 3x
# Then: a separate alert line for the segment (65% recall inside it) vs one national line.
# Two forward splits: train<=2021 -> 2022-24, train<=2024 -> 2025+.
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
y = df[TARGET].astype(int).values; yr = df["year"].values
SEG = {"A: within 5 km of a road": (df["dist_to_road_m"] <= 5000).values, "B: 1,000+ people within 25 km": (df["pop_within_25km"] >= 1000).values}
def lift(yt, s): b = yt.mean(); return average_precision_score(yt, s) / b
def gain_ci(yt, a, b, n=300):
    r = np.random.RandomState(1); g = []
    for _ in range(n):
        i = r.randint(0, len(yt), len(yt))
        if yt[i].sum(): g.append(average_precision_score(yt[i], b[i]) - average_precision_score(yt[i], a[i]))
    return np.mean(g), np.percentile(g, [2.5, 97.5])
def thr65(s, yt): return float(np.sort(s[yt == 1])[int(0.35 * (yt == 1).sum())])

for sname, seg in SEG.items():
    print(f"\n################ Segment {sname} ################")
    for name, trm, vam, tem in (("train<=2021 -> test 2022-24", yr <= 2021, None, (yr >= 2022) & (yr <= 2024)),
                                ("train<=2024 -> test 2025+", yr <= 2024, (yr >= 2022) & (yr <= 2024), yr >= 2025)):
        trs, tes = trm & seg, tem & seg; yt = y[tes]
        print(f"\n[{name}] segment: {seg[tem].mean():.0%} of test fires, {yt.sum():,} big of {len(yt):,} ({yt.mean():.1%}); outside the segment {y[tem & ~seg].mean():.1%} big")
        nat = LGBMClassifier(**P).fit(df.loc[trm, feats], y[trm]); s1 = nat.predict_proba(df.loc[tes, feats])[:, 1]
        spec = LGBMClassifier(**P).fit(df.loc[trs, feats], y[trs]); s2 = spec.predict_proba(df.loc[tes, feats])[:, 1]
        w = np.where(seg[trm], 3.0, 1.0); wt = LGBMClassifier(**P).fit(df.loc[trm, feats], y[trm], sample_weight=w); s3 = wt.predict_proba(df.loc[tes, feats])[:, 1]
        for lab, s in (("1 national model", s1), ("2 specialist (segment only)", s2), ("3 national, segment x3", s3)):
            g = "" if s is s1 else "  gain PR %+.3f (%+.3f to %+.3f)" % ((lambda r: (r[0], r[1][0], r[1][1]))(gain_ci(yt, s1, s)))
            print(f"  {lab:<30} ROC {roc_auc_score(yt, s):.3f} | PR {average_precision_score(yt, s):.3f} | lift {lift(yt, s):.1f}x{g}")
        # separate alert line for the segment
        if vam is not None:
            vs = vam & seg; sv_nat = nat.predict_proba(df.loc[vs, feats])[:, 1]
            t_nat_all = thr65(nat.predict_proba(df.loc[vam, feats])[:, 1], y[vam]); t_seg = thr65(sv_nat, y[vs])
            S = nat.predict_proba(df.loc[tem, feats])[:, 1]; yy = y[tem]; sg = seg[tem]
            one = S >= t_nat_all; two = np.where(sg, S >= t_seg, S >= t_nat_all)
            for lab, fl in (("one national line", one), (f"separate line in segment ({t_seg:.2f})", two)):
                print(f"  {lab:<36} flags {fl.mean():.0%} of fires | catches {(fl & (yy==1)).sum()/yy.sum():.0%} of all big fires | inside segment catches {(fl & (yy==1) & sg).sum()/(yy[sg]==1).sum():.0%}, precision {(fl & (yy==1) & sg).sum()/max((fl & sg).sum(),1):.0%}")
print("\nDone. Paste all of this output back.")
