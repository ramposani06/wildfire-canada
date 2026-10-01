# Honest "known before the report day" score: full model vs model WITHOUT the 4 satellite features,
# with bootstrap ranges. Run in Colab after mounting Drive. Saves a no-satellite model next to the others.
import os, json, warnings, joblib
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")

FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive")
CSV    = os.environ.get("WF_CSV", "unified_dataset_2004_2026_FINAL_v4.csv")
INFO   = "final_model_v14.6_clean_info.json"
SAT    = ["modis_count_early7d", "modis_max_frp_early7d", "viirs_count_early7d", "viirs_max_frp_early7d"]
TARGET, YEAR, LAT, LON = "is_big_fire", "year", "LATITUDE", "LONGITUDE"
TARGET_RECALL, FLAG_TOP, NBOOT = 0.65, 0.15, 500
PARAMS = dict(n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=31, is_unbalance=True,
              random_state=42, verbose=-1)

def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError(name)

feats_all = json.load(open(find(INFO)))["features"]
feats_nosat = [f for f in feats_all if f not in SAT]
print(f"features: full {len(feats_all)} | no satellite {len(feats_nosat)}")

df = pd.read_csv(find(CSV), low_memory=False)
df = df[df[LAT].between(41, 84) & df[LON].between(-142, -52)].dropna(subset=[TARGET]).reset_index(drop=True)
y = df[TARGET].astype(int).values; yr = df[YEAR].values
tr21, va, tr24, te = yr <= 2021, np.isin(yr, [2022, 2023, 2024]), yr <= 2024, yr >= 2025
print(f"train<=2021 {tr21.sum():,} | validation {va.sum():,} | train<=2024 {tr24.sum():,} | test 2025+ {te.sum():,} ({y[te].mean():.3f} big)")

def boot(yt, s, n=NBOOT, seed=1):
    r = np.random.RandomState(seed); a, p = [], []
    for _ in range(n):
        i = r.randint(0, len(yt), len(yt))
        if yt[i].sum() == 0: continue
        a.append(roc_auc_score(yt[i], s[i])); p.append(average_precision_score(yt[i], s[i]))
    return np.percentile(a, [2.5, 97.5]), np.percentile(p, [2.5, 97.5])

def run(label, feats):
    # threshold for ~65% recall from the model trained to 2021, scored on 2022-2024
    m21 = LGBMClassifier(**PARAMS).fit(df.loc[tr21, feats], y[tr21])
    sv = m21.predict_proba(df.loc[va, feats])[:, 1]; yv = y[va]
    thr = np.sort(sv[yv == 1])[int((1 - TARGET_RECALL) * (yv == 1).sum())]
    # final model trained to 2024, scored on 2025+
    m24 = LGBMClassifier(**PARAMS).fit(df.loc[tr24, feats], y[tr24])
    s = m24.predict_proba(df.loc[te, feats])[:, 1]; yt = y[te]
    ra, pa = boot(yt, s)
    print(f"\n== {label} ({len(feats)} features) ==")
    print(f"Validation 2022-24: ROC-AUC {roc_auc_score(yv, sv):.3f} | PR-AUC {average_precision_score(yv, sv):.3f}")
    print(f"TEST 2025+ : ROC-AUC {roc_auc_score(yt, s):.3f} ({ra[0]:.3f}-{ra[1]:.3f}) | PR-AUC {average_precision_score(yt, s):.3f} ({pa[0]:.3f}-{pa[1]:.3f})")
    for nm, mk in (("2025", yr[te] == 2025), ("2026", yr[te] == 2026)):
        print(f"  {nm}: n={mk.sum():,} ROC-AUC {roc_auc_score(yt[mk], s[mk]):.3f} | PR-AUC {average_precision_score(yt[mk], s[mk]):.3f}")
    fl = s >= thr
    print(f"Rule A (threshold {thr:.3f} from validation): flags {fl.mean():.0%}, catches {(fl & (yt==1)).sum()/yt.sum():.0%}, precision {(fl & (yt==1)).sum()/fl.sum():.0%}")
    cut = np.quantile(s, 1 - FLAG_TOP); fl2 = s >= cut
    print(f"Rule B (top 15%, cut {cut:.3f}): catches {(fl2 & (yt==1)).sum()/yt.sum():.0%}, precision {(fl2 & (yt==1)).sum()/fl2.sum():.0%}")
    order = np.argsort(-s)
    print("Top-k capture:", {f"{int(q*100)}%": f"{yt[order[:int(q*len(s))]].sum()/yt.sum():.0%}" for q in (.05, .10, .15, .20, .25)})
    print(f"Average raw score {s.mean():.3f} vs actual rate {yt.mean():.3f}")
    return thr, m24, feats

run("FULL model, same-day satellite included", feats_all)
thr, m24, f = run("NO-SATELLITE model (known before the report day)", feats_nosat)

# save the no-satellite model, trained on all years, with its own info file (does not touch the existing files)
final = LGBMClassifier(**PARAMS).fit(df[feats_nosat], y)
out = os.path.join(FOLDER, "final_model_v14.6_nosat_allyears.pkl"); joblib.dump(final, out)
json.dump({"features": feats_nosat, "threshold_recall": float(thr), "flag_top_fraction": FLAG_TOP,
           "note": "no satellite features; raw scores, not probabilities; calibrator not fitted for this model"},
          open(os.path.join(FOLDER, "final_model_v14.6_nosat_info.json"), "w"), indent=1)
print("\nSaved:", out)
print("Done. Paste all of this output back.")
