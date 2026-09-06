"""t35_d_invariance.py -- the prediction the paper had not tested.

Theorem 4 says the risk depends on the budget only through u = m/M, so the
optimal budget is free of d. Corollary 5 says that appending columns inside
existing blocks leaves the block rule untouched and costs the all-candidate
forest the whole collision term. Neither had been tested, because every earlier
experiment moved d and M together.

This moves d alone. Starting from the fifteen augmented columns, we append
copies of existing columns perturbed by a small amount of noise, so each copy
falls inside the block its parent belongs to and M is held fixed by
construction. d then runs 15, 20, 25, 30, 40 while M does not move.

Three things are read off the result:

  1. the optimal u should not move with d;
  2. Dynamic ET, which sets m = M, should not degrade with d;
  3. the all-candidate forest, which sets m = d, should degrade, because its
     collision probability rises towards one as columns are added.

The noise level is chosen so a copy stays inside its parent's block at the
threshold the method uses; the realised M is reported for every setting so the
construction can be checked rather than assumed.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from scipy.stats import wilcoxon
from t4_clean_sweep import load, augment, blocks, BASE

MAX_TR, MAX_TE, SEED, NT, TAU = 30_000, 15_000, 42, 1000, 0.75
WIDTHS = [15, 20, 25, 30, 40]
UGRID = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00]
HERE = Path(__file__).resolve().parents[1]


def widen(A, d_target, rng, noise=0.02):
    """Append near-copies of existing columns so M is unchanged and d grows."""
    if d_target <= A.shape[1]:
        return A
    extra = d_target - A.shape[1]
    idx = rng.integers(0, A.shape[1], extra)
    cols = []
    for j in idx:
        c = A[:, j]
        sd = c.std() if c.std() > 0 else 1.0
        cols.append(c + rng.normal(0, noise * sd, len(c)))
    return np.hstack([A, np.column_stack(cols)])


def main():
    d = load(); d['q'] = d['date'].dt.to_period('Q')
    qs = sorted(d['q'].unique()); rows = []
    path = HERE / 'results' / 'd_invariance.csv'
    for i in range(4, len(qs)):
        tr = d[d['q'].isin(qs[i-4:i])]; te = d[d['q'] == qs[i]]
        if len(tr) < 500 or len(te) < 50:
            continue
        rng = np.random.default_rng(SEED)
        if len(tr) > MAX_TR: tr = tr.iloc[rng.choice(len(tr), MAX_TR, replace=False)]
        if len(te) > MAX_TE: te = te.iloc[rng.choice(len(te), MAX_TE, replace=False)]
        A0, A0te = augment(tr[BASE].values), augment(te[BASE].values)
        ytr, yte = tr['IV'].values, te['IV'].values
        fold = str(qs[i])

        for D in WIDTHS:
            r2 = np.random.default_rng(SEED + D)
            X = widen(A0, D, r2)
            r2 = np.random.default_rng(SEED + D)          # same columns on test
            Xte = widen(A0te, D, r2)
            dd = X.shape[1]; M = blocks(X, TAU)
            for u in UGRID:
                m = max(1, min(dd, int(round(u * M))))
                p = ExtraTreesRegressor(n_estimators=NT, max_depth=None,
                        min_samples_leaf=1, max_features=m/dd, bootstrap=False,
                        n_jobs=-1, random_state=SEED).fit(X, ytr).predict(Xte)
                rows.append(dict(fold=fold, d=dd, M=M, u=u, m=m, arm='grid',
                                 r2=r2_score(yte, p)))
            # the two rules, as they would actually be set
            for name, m in (('Dynamic ET', M), ('ET-aug', dd)):
                p = ExtraTreesRegressor(n_estimators=NT, max_depth=None,
                        min_samples_leaf=1, max_features=m/dd, bootstrap=False,
                        n_jobs=-1, random_state=SEED).fit(X, ytr).predict(Xte)
                rows.append(dict(fold=fold, d=dd, M=M, u=m/M, m=m, arm=name,
                                 r2=r2_score(yte, p)))
            print(f'{fold} d={dd} M={M}', flush=True)
            pd.DataFrame(rows).to_csv(path, index=False)

    t = pd.DataFrame(rows)
    print(f'\n{len(t)} rows over {t.fold.nunique()} folds -> {path}\n')
    print('realised block count by width (M must not move):')
    print(t.groupby('d').M.agg(['mean','min','max']).to_string(float_format=lambda x: f'{x:.2f}'))
    g = t[t.arm == 'grid']
    print('\nmean R2 by (d, u); the optimum should not move across rows:')
    piv = g.pivot_table(index='d', columns='u', values='r2', aggfunc='mean')
    print(piv.to_string(float_format=lambda x: f'{x:.4f}'))
    print('\nargmax u by width:', {int(d_): float(r.idxmax()) for d_, r in piv.iterrows()})
    print('\nthe two rules as d grows:')
    print(t[t.arm != 'grid'].pivot_table(index='d', columns='arm', values='r2',
            aggfunc='mean').to_string(float_format=lambda x: f'{x:.4f}'))
    piv2 = t[t.arm != 'grid'].pivot_table(index=['d','fold'], columns='arm', values='r2')
    print()
    for D in WIDTHS:
        try: dd_ = (piv2.loc[D, 'Dynamic ET'] - piv2.loc[D, 'ET-aug']).dropna()
        except KeyError: continue
        if len(dd_) < 4: continue
        p = wilcoxon(dd_, alternative='greater').pvalue
        print(f'  d={D:2}  Dynamic ET - ET-aug = {dd_.mean():+.4f}  '
              f'{int((dd_>0).sum())}/{len(dd_)}  p={p:.4g}')


if __name__ == '__main__':
    main()
