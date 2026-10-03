# Lists the lightning files on Drive (CanCPLD etc.) and peeks inside, so the lightning feature test can be written for the real format.
import os, re, glob
FOLDER = os.environ.get("WF_FOLDER", "/content/drive/MyDrive")
hits = []
for root, _, files in os.walk(FOLDER):
    for f in files:
        p = os.path.join(root, f)
        if re.search(r"cancpld|lightning|cg_|strike|glm", p, re.I):
            hits.append((p, os.path.getsize(p) / 1e6))
hits.sort()
print(f"{len(hits)} lightning-related files under {FOLDER}")
by_dir = {}
for p, mb in hits: by_dir.setdefault(os.path.dirname(p), []).append((os.path.basename(p), mb))
for d, fl in by_dir.items():
    exts = {}
    for n, mb in fl: exts.setdefault(os.path.splitext(n)[1].lower(), []).append(mb)
    print(f"\n{d}: {len(fl)} files | " + ", ".join(f"{k or '(none)'} x{len(v)} ({sum(v):.0f} MB)" for k, v in exts.items()))
    print("   examples:", [n for n, _ in fl[:6]])
print("\n=== peek inside one file of each type ===")
seen = set()
for p, mb in hits:
    ext = os.path.splitext(p)[1].lower()
    if ext in seen or mb > 2000: continue
    seen.add(ext)
    print(f"\n--- {p} ({mb:.1f} MB) ---")
    try:
        if ext in (".csv", ".txt"):
            import pandas as pd
            d = pd.read_csv(p, nrows=5, low_memory=False); print(d.columns.tolist()); print(d.head(3).to_string())
        elif ext in (".nc", ".nc4"):
            import xarray as xr
            ds = xr.open_dataset(p); print(ds)
        elif ext == ".zip":
            import zipfile; z = zipfile.ZipFile(p); print(z.namelist()[:10])
        elif ext in (".parquet",):
            import pandas as pd
            d = pd.read_parquet(p); print(d.columns.tolist(), len(d)); print(d.head(3).to_string())
        elif ext in (".pkl",):
            print("pickle file (skipped)")
        else:
            print("first bytes:", open(p, "rb").read(200))
    except Exception as e:
        print("could not open:", str(e)[:200])
print("\nDone. Paste all of this output back.")
