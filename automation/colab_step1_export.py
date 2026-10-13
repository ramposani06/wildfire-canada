# STEP 1 for automation (run in Colab). Does two things:
#   A) copies the v14.7 model files from your Drive into the repo (models/) and pushes them
#   B) checks whether the road and population source files exist on your Drive
# Needs a Colab Secret named GITHUB_TOKEN (a token with "Contents: write" on this repo only).
# The token is never printed or written to a file. Delete the token on GitHub when you are done.
import os, glob, shutil, subprocess, json
from google.colab import drive, userdata
drive.mount("/content/drive")

DRIVE = "/content/drive/MyDrive"
REPO_URL = "github.com/ramposani06/wildfire-canada"
MODEL_FILES = ["final_model_v14.7_allyears.pkl", "final_model_v14.7_info.json"]

def find(name):
    hits = glob.glob(f"{DRIVE}/**/{name}", recursive=True)
    return hits[0] if hits else None

# ---------- B) what is on Drive for roads and population ----------
print("=== SOURCE FILES ON DRIVE ===")
roads = glob.glob(f"{DRIVE}/**/*.shp", recursive=True)
gpkg = glob.glob(f"{DRIVE}/**/*.gpkg", recursive=True)
for label, files in [("road shapefiles (.shp)", roads), ("geopackages (.gpkg, population?)", gpkg)]:
    total = sum(os.path.getsize(f) for f in files) / 1e9
    print(f"{label}: {len(files)} files, {total:.2f} GB")
    for f in files[:15]:
        print("   ", f.replace(DRIVE, "MyDrive"), f"{os.path.getsize(f)/1e6:.0f} MB")

# ---------- A) copy model files into the repo and push ----------
tok = userdata.get("GITHUB_TOKEN")
os.chdir("/content")
subprocess.run("rm -rf wildfire-canada", shell=True)
subprocess.run(["git", "clone", "-q", f"https://x-access-token:{tok}@{REPO_URL}.git"], check=True)
os.chdir("/content/wildfire-canada")
missing = []
for name in MODEL_FILES:
    src = find(name)
    if src is None:
        missing.append(name); continue
    shutil.copy(src, f"models/{name}")
    print("copied", name, f"({os.path.getsize(src)/1e6:.2f} MB)")
if missing:
    print("NOT FOUND on Drive:", missing, "-> nothing pushed")
else:
    info = json.load(open("models/final_model_v14.7_info.json"))
    print("v14.7 features:", len(info["features"]), "| alert line:", round(info["threshold_recall"], 4))
    subprocess.run(["git", "config", "user.email", "ram.posani06@gmail.com"])
    subprocess.run(["git", "config", "user.name", "ramposani06"])
    subprocess.run(["git", "add", "models/final_model_v14.7_allyears.pkl", "models/final_model_v14.7_info.json"], check=True)
    r = subprocess.run(["git", "commit", "-qm", "Add frozen v14.7 model files"], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    p = subprocess.run(["git", "push", "-q", "origin", "HEAD:main"], capture_output=True, text=True)
    print("pushed" if p.returncode == 0 else "push failed: " + p.stderr.replace(tok, "***"))
