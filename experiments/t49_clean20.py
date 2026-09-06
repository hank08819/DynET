"""t49_clean20.py -- 无泄漏的十二列基础, 六项与八项两种增广的对照 (18 对 20)."""
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from scipy.stats import wilcoxon
from t4_clean_sweep import load, BASE
from dynET import base12_leakfree, augment_clean18, augment_clean20, block_count

MAX_TR, MAX_TE, SEED, NT = 30_000, 15_000, 42, 1000
HERE = Path(__file__).resolve().parents[1]

d = load(); d['q'] = d['date'].dt.to_period('Q')
qs = sorted(d['q'].unique()); rows = []
for i in range(4, len(qs)):
    tr = d[d['q'].isin(qs[i-4:i])]; te = d[d['q'] == qs[i]]
    if len(tr) < 500 or len(te) < 50: continue
    rng = np.random.default_rng(SEED)
    if len(tr) > MAX_TR: tr = tr.iloc[rng.choice(len(tr), MAX_TR, replace=False)]
    if len(te) > MAX_TE: te = te.iloc[rng.choice(len(te), MAX_TE, replace=False)]
    B12, B12te = base12_leakfree(tr[BASE].values), base12_leakfree(te[BASE].values)
    ytr, yte = tr['IV'].values, te['IV'].values
    line = []
    for nm, fn in (('clean18', augment_clean18), ('clean20', augment_clean20)):
        A, Ate = fn(B12), fn(B12te)
        dd = A.shape[1]; M = block_count(A)
        p = ExtraTreesRegressor(n_estimators=NT, max_depth=None, min_samples_leaf=1,
                max_features=M/dd, bootstrap=False, n_jobs=-1,
                random_state=SEED).fit(A, ytr).predict(Ate)
        rows.append(dict(fold=str(qs[i]), regime=nm, d=dd, M=M, r2=r2_score(yte, p)))
        line.append(f'{nm} M={M}/{dd} {rows[-1]["r2"]:.4f}')
    print(f'{qs[i]}  ' + '  '.join(line), flush=True)
t = pd.DataFrame(rows); t.to_csv(HERE/'results'/'clean20.csv', index=False)
piv = t.pivot(index='fold', columns='regime', values='r2')
print('\n均值:'); print(t.groupby('regime')[['r2','M']].mean().round(4).to_string())
print(f"\n20 列 - 18 列: {(piv.clean20-piv.clean18).mean():+.4f}  "
      f"{int((piv.clean20>piv.clean18).sum())}/{len(piv)} 折  "
      f"p={wilcoxon(piv.clean20, piv.clean18)[1]:.4f}")
