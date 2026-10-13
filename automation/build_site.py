"""Build the one-link project site (GitHub Pages) from files already in the repo.

Pages made:
  index.html       README (landing page) + links
  model_card.html  docs/model_card.md
  dashboard.html   monitoring/dashboard.html (older v14.6 numbers, labelled as such)
  status.json      when the site was built and from which commit
Run:  python automation/build_site.py   (writes ./site)
"""
import json
import os
import re
import shutil
from datetime import datetime, timezone

import markdown

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "site")

CSS = """
:root{--bg:#fff;--fg:#1d2330;--mut:#5b6475;--line:#dde1e8;--acc:#b3261e;--code:#f3f5f8}
@media (prefers-color-scheme:dark){:root{--bg:#14171d;--fg:#e6e9ef;--mut:#9aa3b2;--line:#2b303b;--acc:#ff8a80;--code:#1d222b}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:860px;margin:0 auto;padding:16px 16px 64px}
nav{border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:16px;flex-wrap:wrap;position:sticky;top:0;background:var(--bg)}
nav a{color:var(--fg);text-decoration:none;font-weight:600} nav a:hover{color:var(--acc)}
a{color:var(--acc)} img{max-width:100%;height:auto}
table{border-collapse:collapse;display:block;overflow-x:auto;max-width:100%}
th,td{border:1px solid var(--line);padding:6px 10px;text-align:left}
code,pre{background:var(--code);border-radius:4px} code{padding:1px 4px} pre{padding:12px;overflow-x:auto}
footer{color:var(--mut);font-size:14px;margin-top:40px;border-top:1px solid var(--line);padding-top:12px}
"""

NAV = ('<nav><a href="index.html">Overview</a><a href="model_card.html">Model card</a>'
       '<a href="dashboard.html">Monitoring (older)</a><a href="status.json">Status</a></nav>')


def page(title, body, stamp):
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{title}</title><style>{CSS}</style></head><body>{NAV}<main>{body}'
            f'<footer>{stamp}</footer></main></body></html>')


def md_to_html(text):
    return markdown.markdown(text, extensions=["tables", "fenced_code", "toc", "sane_lists"])


def fix_links(html):
    """Point links between the markdown files at the built pages."""
    html = html.replace('href="docs/model_card.md', 'href="model_card.html')
    html = re.sub(r'href="docs/archive/[^"]*"', 'href="https://github.com/ramposani06/wildfire-canada/tree/main/docs/archive"', html)
    html = html.replace('src="docs/img/', 'src="img/')
    html = html.replace('href="monitoring/dashboard.html"', 'href="dashboard.html"')
    html = html.replace('href="monitoring/"', 'href="dashboard.html"')
    return html


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    shutil.copytree(os.path.join(ROOT, "docs", "img"), os.path.join(OUT, "img"))

    sha = os.environ.get("GITHUB_SHA", "local")[:7]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    stamp = (f'Built {now} from commit {sha}. '
             f'<a href="https://github.com/ramposani06/wildfire-canada">Source on GitHub</a>')

    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    card = open(os.path.join(ROOT, "docs", "model_card.md"), encoding="utf-8").read()
    open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(
        page("Canada big-fire prediction", fix_links(md_to_html(readme)), stamp))
    open(os.path.join(OUT, "model_card.html"), "w", encoding="utf-8").write(
        page("Model card", fix_links(md_to_html(card)), stamp))

    dash = open(os.path.join(ROOT, "monitoring", "dashboard.html"), encoding="utf-8").read()
    banner = ('<div style="background:#fff3cd;color:#5c4400;padding:8px 16px;font:14px system-ui">'
              'This dashboard still shows the older v14.6 numbers. The current numbers (v14.7) are in the '
              '<a href="model_card.html">model card</a>.</div>')
    dash = re.sub(r"(<body[^>]*>)", r"\1" + banner, dash, count=1) if "<body" in dash else banner + dash
    open(os.path.join(OUT, "dashboard.html"), "w", encoding="utf-8").write(dash)

    json.dump({"built_utc": now, "commit": sha, "model": "v14.7 (single LightGBM, 22 features)",
               "pages": ["index.html", "model_card.html", "dashboard.html"]},
              open(os.path.join(OUT, "status.json"), "w"), indent=2)
    open(os.path.join(OUT, ".nojekyll"), "w").close()
    print("site built:", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
