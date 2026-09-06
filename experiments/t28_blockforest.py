"""t28_blockforest.py -- Dynamic ET against Block Forests, leak-free, paper protocol.

Block Forests \\citep{hornung2019block} is the established forest for grouped
covariates and is the closest published method to the rule proposed here: both
let block structure govern the split-candidate budget. They differ in where the
blocks come from -- an a-priori semantic grouping there, an inferred correlation
grouping here -- and in how the per-block budget is set. Trees, features, folds
and training samples are held fixed and only those two things vary.

The earlier run of this comparison used the twelve-feature vector, four columns
of which are a function of the target. This repeats it on the leak-free features
and at the manuscript's own protocol: 1,000 trees, 30,000 training rows, 15,000
test rows, the sixteen AAPL quarters.
"""
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import wilcoxon
from t4_clean_sweep import load, augment, blocks, BASE
from blocked_et import BlockedExtraTrees

MAX_TR, MAX_TE, SEED, NT, TAU = 30_000, 15_000, 42, 1000, 0.75
# a-priori semantic grouping of the fifteen leak-free columns: contract geometry,
# the liquidity proxy, and the interaction terms built from them.
SEM = np.array([0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2])
HERE = Path(__file__).resolve().parents[1]


def corr_blocks(X, tau=TAU):
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
    lab = np.array([f(i) for i in range(d)])
    _, inv = np.unique(lab, return_inverse=True)
    return inv


def main():
    d = load(); d['q'] = d['date'].dt.to_period('Q')
    qs = sorted(d['q'].unique()); rows = []
    path = HERE / 'results' / 'blockforest.csv'
    for i in range(4, len(qs)):
        tr = d[d['q'].isin(qs[i-4:i])]; te = d[d['q'] == qs[i]]
        if len(tr) < 500 or len(te) < 50:
            continue
        rng = np.random.default_rng(SEED)
        if len(tr) > MAX_TR: tr = tr.iloc[rng.choice(len(tr), MAX_TR, replace=False)]
        if len(te) > MAX_TE: te = te.iloc[rng.choice(len(te), MAX_TE, replace=False)]
        Xtr, Xte = augment(tr[BASE].values), augment(te[BASE].values)
        ytr, yte = tr['IV'].values, te['IV'].values
        dd = Xtr.shape[1]
        M = blocks(Xtr)
        cb = corr_blocks(Xtr)
        sem = np.unique(SEM, return_inverse=True)[1]
        # Block Forests draws m_j candidates from block j; the standard choice is
        # the square root of the block's size, as in the reference implementation.
        mv_sem = np.array([max(1, int(np.sqrt((sem == b).sum())))
                           for b in range(sem.max() + 1)])
        mv_cor = np.array([max(1, int(np.sqrt((cb == b).sum())))
                           for b in range(cb.max() + 1)])
        fold = str(qs[i])

        arms = {}
        t0 = time.perf_counter()
        arms['Dynamic ET'] = ExtraTreesRegressor(
            n_estimators=NT, max_depth=None, min_samples_leaf=1,
            max_features=M / dd, bootstrap=False, n_jobs=-1,
            random_state=SEED).fit(Xtr, ytr).predict(Xte)
        arms['ET-aug'] = ExtraTreesRegressor(
            n_estimators=NT, max_depth=None, min_samples_leaf=1,
            max_features=1.0, bootstrap=False, n_jobs=-1,
            random_state=SEED).fit(Xtr, ytr).predict(Xte)
        arms['BF-varsel'] = BlockedExtraTrees(
            NT, mode=2, mvec=mv_sem, blocks=sem).fit(Xtr, ytr).predict(Xte)
        arms['BF-default'] = BlockedExtraTrees(
            NT, mode=4, mvec=mv_sem, blocks=sem).fit(Xtr, ytr).predict(Xte)
        arms['BF-corr'] = BlockedExtraTrees(
            NT, mode=4, mvec=mv_cor, blocks=cb).fit(Xtr, ytr).predict(Xte)

        line = []
        for name, p in arms.items():
            p = np.asarray(p, dtype=float)
            rows.append(dict(fold=fold, model=name, M_blocks=M, d=dd,
                             r2=r2_score(yte, p), mae=mean_absolute_error(yte, p)))
            line.append(f'{name} {rows[-1]["r2"]:.4f}')
        print(f'{fold}  M={M}/{dd}  ' + '  '.join(line) +
              f'  ({time.perf_counter()-t0:.0f}s)', flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)

    t = pd.DataFrame(rows)
    g = t.groupby('model')[['r2', 'mae']].mean().sort_values('r2', ascending=False)
    print(f'\n{len(t)} fits over {t.fold.nunique()} folds -> {path}\n')
    print(g.to_string(float_format=lambda x: f'{x:.4f}'))
    piv = t.pivot(index='fold', columns='model', values='r2')
    print()
    for m in piv.columns:
        if m == 'Dynamic ET':
            continue
        dd_ = (piv['Dynamic ET'] - piv[m]).dropna()
        p = wilcoxon(dd_, alternative='greater').pvalue
        print(f'  vs {m:<12} dR2={dd_.mean():+.4f}  {int((dd_>0).sum())}/{len(dd_)}'
              f'  p={p:.4g}')


if __name__ == '__main__':
    main()
