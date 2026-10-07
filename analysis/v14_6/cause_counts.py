# Count all fires by cause (natural / human / other) and big-fire rate. Run in Colab.
import pandas as pd
D = '/content/drive/MyDrive/'
df = pd.read_csv(D + 'unified_dataset_2004_2026_FINAL_v4.csv', low_memory=False)
print('rows:', len(df))

c = df['CAUSE'].astype(str).str.strip().str.upper().str[0]
df['cause_group'] = c.map({'N': 'Natural (lightning)', 'H': 'Human'}).fillna('Other / unknown')

# big fire flag: try common column names
big = None
for col in ['is_big', 'BIG', 'big_fire', 'target', 'TARGET']:
    if col in df.columns:
        big = df[col]; break
if big is None:
    sz = [x for x in df.columns if 'SIZE' in x.upper() or 'HA' == x.upper()]
    if sz: big = (pd.to_numeric(df[sz[0]], errors='coerce') > 100).astype(int)
df['big'] = big

def table(g):
    t = df.groupby(g).agg(fires=('big', 'size'), big=('big', 'sum'))
    t['% of fires'] = (100 * t.fires / t.fires.sum()).round(1)
    t['% big'] = (100 * t.big / t.fires).round(1)
    return t

print('\n=== ALL FIRES BY CAUSE ===')
print(table('cause_group'))
print('\n=== RAW CAUSE CODES ===')
print(df['CAUSE'].value_counts(dropna=False).head(15))
if 'CAUSE2' in df.columns:
    print('\n=== CAUSE2 ===')
    print(df['CAUSE2'].value_counts(dropna=False).head(15))
if 'PRESCRIBED' in df.columns:
    print('\n=== PRESCRIBED ===')
    print(df['PRESCRIBED'].value_counts(dropna=False).head(10))
yc = [x for x in df.columns if x.upper() in ('YEAR_CLEAN', 'YEAR')]
if yc:
    print('\n=== BY YEAR (fires) ===')
    print(pd.crosstab(df[yc[0]], df['cause_group']))
