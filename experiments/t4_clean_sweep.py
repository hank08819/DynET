"""t4_clean_sweep.py -- the sweep again, with the leaking features removed.

t1 ran on the manuscript's twelve features, which include the Greeks. t3 shows
those carry the target: dropping them takes R^2 from 0.966 to 0.899, and a vendor
obtains them by differentiating the pricing formula at the IV it solved for. In
that setting a rule that restricts split candidates can only get in the way of a
near-decisive feature, so t1 could not have found the block rule useful whether
or not it is.

This repeats the sweep on the feature set of controls/dynet_vs_blockforest_clean.py,
which drops the Greeks and the quote midpoint and rebuilds the interactions
without them. The existing control shows Dynamic ET beating ET on those features
at one budget, 13 of 16 folds. The open question it leaves is whether the inferred
M/d beats every fixed rate, or merely the one the manuscript compared against.

Protocol follows the clean control exactly: 150 trees, 20,000 train, 10,000 test,
tau = 0.75, seed 42.
"""
import sys, time
from pathlib import Path
import os
DATA_DIR = os.environ.get('DYNET_DATA', os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), 'data'))
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error

D = Path(DATA_DIR)
PANEL = D / 'aapl_2016_2020.csv'
N_TREES, N_TR, N_TE, TAU, SEED = 150, 20_000, 10_000, 0.75, 42
BASE = ['m', 'logm', 'T', 'K', 'S', 'is_call', 'spr']
GRID = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
HERE = Path(__file__).resolve().parents[1]


def load():                                     # controls/dynet_vs_blockforest_clean.py
    raw = pd.read_csv(PANEL, low_memory=False, on_bad_lines='skip')
    raw.columns = [c.strip() for c in raw.columns]
    n = lambda c: pd.to_numeric(raw[c], errors='coerce')
    dates = pd.to_datetime(raw['[QUOTE_DATE]'], errors='coerce'); fr = []
    for t, ty in (('C', 0), ('P', 1)):
        fr.append(pd.DataFrame({'date': dates, 'S': n('[UNDERLYING_LAST]'),
            'K': n('[STRIKE]'), 'T': n('[DTE]') / 365., 'IV': n(f'[{t}_IV]'),
            'bid': n(f'[{t}_BID]'), 'ask': n(f'[{t}_ASK]'), 'is_call': 1 - ty}))
    d = pd.concat(fr, ignore_index=True).dropna()
    d = d[(d.IV > 0.01) & (d.IV < 4) & (d['T'] > 1/365) & (d.K > 0) & (d.ask >= d.bid)]
    d['m'] = d.S / d.K; d['logm'] = np.log(d.m); d['spr'] = d.ask - d.bid
    return d[d.m.between(0.5, 2.0)].sort_values('date').reset_index(drop=True)


def augment(X):                                 # interactions without Greeks or quote
    m, lm, t, sp = X[:, 0], X[:, 1], X[:, 2], X[:, 6]
    st = np.where(t > 1e-8, t, 1e-8)
    return np.hstack([X, np.column_stack([m*t, lm**2, m*sp, sp/st, m**2, t**2, lm*t, sp*t])])


def blocks(X, tau=TAU):
    C = np.abs(np.corrcoef(np.where(np.isfinite(X), X, 0.).T)); np.fill_diagonal(C, 0.)
    d = C.shape[0]; par = list(range(d))
    def f(a):
        while par[a] != a: par[a] = par[par[a]]; a = par[a]
        return a
    for i in range(d):
        for j in range(i + 1, d):
            if C[i, j] > tau:
                a, b = f(i), f(j)
                if a != b: par[b] = a
    return len({f(i) for i in range(d)})


def et(mf):
    return ExtraTreesRegressor(n_estimators=N_TREES, max_depth=None, min_samples_leaf=1,
                               max_features=mf, bootstrap=False, n_jobs=-1,
                               random_state=SEED)


def main():
    d = load(); d['q'] = d['date'].dt.to_period('Q')
    qs = sorted(d['q'].unique()); rows = []
    path = HERE / 'results' / 'clean_sweep.csv'
    for i in range(4, len(qs)):
        tr = d[d['q'].isin(qs[i-4:i])]; te = d[d['q'] == qs[i]]
        if len(tr) < 500 or len(te) < 50:
            continue
        rng = np.random.default_rng(SEED)
        if len(tr) > N_TR: tr = tr.iloc[rng.choice(len(tr), N_TR, replace=False)]
        if len(te) > N_TE: te = te.iloc[rng.choice(len(te), N_TE, replace=False)]
        B_tr, B_te = tr[BASE].values, te[BASE].values
        A_tr, A_te = augment(B_tr), augment(B_te)
        ytr, yte = tr['IV'].values, te['IV'].values
        M, dd = blocks(A_tr), A_tr.shape[1]
        fold = str(qs[i])
        print(f"\nFold {fold}  train={len(tr):,}  test={len(te):,}  "
              f"M={M}/{dd}={M/dd:.3f}", flush=True)
        arms = [('base-7', B_tr, B_te, 1.0), ('DynET', A_tr, A_te, M / dd)]
        arms += [(f'aug@{g:.2f}', A_tr, A_te, g) for g in GRID]
        for name, A, Bt, mf in arms:
            t0 = time.perf_counter()
            p = et(mf).fit(A, ytr).predict(Bt)
            rows.append(dict(fold=fold, arm=name, d=A.shape[1], max_features=mf,
                             M_blocks=M, r2=r2_score(yte, p),
                             mae=mean_absolute_error(yte, p)))
            print(f"  {name:<10} d={A.shape[1]:<3} mf={mf:.3f}  R2={rows[-1]['r2']:.4f}"
                  f"  ({time.perf_counter()-t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(path, index=False)
    t = pd.DataFrame(rows)
    print(f'\n{len(t)} fits over {t.fold.nunique()} folds -> {path}\n')
    print(t.groupby('arm').r2.agg(['mean', 'std', 'count'])
           .sort_values('mean', ascending=False)
           .to_string(float_format=lambda x: f'{x:.4f}'))


if __name__ == '__main__':
    main()
