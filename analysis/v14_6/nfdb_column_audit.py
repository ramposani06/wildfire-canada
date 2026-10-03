# Which raw NFDB columns could be known when a fire is first reported? (run in Colab after mounting Drive)
# Classifies each column: SAFE / INVESTIGATE / LEAKAGE, then looks at the INVESTIGATE ones in detail.
# It does NOT add anything to the model. It only reports.
import os, re, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 80)

FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive")
RAW = os.environ.get("WF_RAW", "NFDB_point_20260811.txt")
def find(name):
    for root, _, files in os.walk(FOLDER):
        if name in files: return os.path.join(root, name)
    raise FileNotFoundError(name)

try:  df = pd.read_csv(find(RAW), low_memory=False)
except UnicodeDecodeError: df = pd.read_csv(find(RAW), low_memory=False, encoding="latin-1")
print(f"{RAW}: {len(df):,} rows, {df.shape[1]} columns\n")

SAFE = {"NFDBFIREID", "SRC_AGENCY", "LATITUDE", "LONGITUDE", "REP_DATE", "YEAR", "MONTH", "DAY", "NAT_PARK", "PROTZONE"}
LEAK_WORDS = ["SIZE", "CALC_HA", "AREA", "OUT_", "_OUT", "END", "CONTAIN", "CONTROL", "DURATION", "PERIM", "BURN"]
INVEST_WORDS = ["CAUSE", "RESPONSE", "ATTK", "ATTACK", "DISC", "REPORT", "FIRE_TYPE", "TYPE", "STATUS", "PRESCRIBED", "OBS", "NAME", "INFO", "FIRE_ID"]
def klass(c):
    u = c.upper()
    if u in SAFE: return "SAFE (check)"
    if any(w in u for w in LEAK_WORDS): return "LEAKAGE"
    if any(w in u for w in INVEST_WORDS): return "INVESTIGATE"
    return "UNKNOWN"

rows = []
for c in df.columns:
    s = df[c]
    rows.append([c, klass(c), str(s.dtype), f"{s.isna().mean():.0%}", s.nunique(), str(s.dropna().iloc[0])[:30] if s.notna().any() else ""])
print(pd.DataFrame(rows, columns=["column", "class", "dtype", "missing", "unique", "example"]).to_string(index=False))

yrs = pd.to_numeric(df.get("YEAR"), errors="coerce")
size = next((c for c in df.columns if c.upper() == "SIZE_HA"), None)
big = (pd.to_numeric(df[size], errors="coerce") > 100) if size else None

print("\n" + "=" * 70 + "\nINVESTIGATE columns: values, missing by recent year, big-fire rate\n" + "=" * 70)
for c in [c for c in df.columns if klass(c) in ("INVESTIGATE", "UNKNOWN")]:
    s = df[c]
    print(f"\n--- {c} ({s.dtype}) ---")
    if s.nunique() <= 25:
        t = pd.DataFrame({"fires": s.value_counts(dropna=False)})
        if big is not None: t["big_rate"] = big.groupby(s.fillna("(missing)")).mean().reindex(t.index.fillna("(missing)")).values.round(3)
        print(t.head(25).to_string())
    else:
        print("sample values:", s.dropna().astype(str).unique()[:6].tolist())
    if yrs is not None:
        m = s.isna().groupby(yrs).mean()
        print("share missing by year (2018-2025):", {int(k): f"{v:.0%}" for k, v in m.items() if 2018 <= k <= 2025})

# timing of attack / discovery dates versus the report date
rep = pd.to_datetime(df.get("REP_DATE"), errors="coerce")
for c in df.columns:
    if any(w in c.upper() for w in ("ATTK", "ATTACK", "DISC", "OUT_DATE", "END")) and "DATE" in c.upper():
        d = pd.to_datetime(df[c], errors="coerce")
        gap = (d - rep).dt.days
        ok = gap.between(-30, 400)
        print(f"\nTiming: {c} minus REP_DATE (days) over {ok.sum():,} fires with both: "
              f"median {gap[ok].median():.0f}, share same day {(gap[ok] == 0).mean():.0%}, share before report {(gap[ok] < 0).mean():.0%}")
print("\nDone. Paste all of this output back.")
