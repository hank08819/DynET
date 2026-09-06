"""t31_tuned_all.py -- the method as proposed, on all five underlyings.

Dynamic ET is proposed with its own settings chosen on the training window, so
the form evaluated across the five panels should be that one. The search is the
same as in t18 -- forty configurations over tau, the leaf size, the depth cap
and the split minimum, drawn at random and scored on a temporal inner split of
the training window, the test quarter never read -- and the winner is refit on
the whole window at full size.

The untuned form is run alongside on the same folds, so the price of the search
is visible and the two can be compared pairwise.
"""
import sys, time, warnings, itertools
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import wilcoxon
from t4_clean_sweep import augment, BASE
from t24_underlyings import load, PANELS, DATA

MAX_TR, MAX_TE, SEED = 30_000, 15_000, 42
N_IN, N_FULL, BUDGET = 300, 1000, 40
GRID = [dict(tau=t, leaf=l, depth=d, split=s)
        for t, l, d, s in itertools.product(
            [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
            [1, 2, 3, 5], [None, 24], [2, 5])]
HERE = Path(__file__).resolve().parents[1]


def blocks_at(X, tau):
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


def et(mf, n, **kw):
    return ExtraTreesRegressor(n_estimators=n, max_features=mf, bootstrap=False,
                               n_jobs=-1, random_state=SEED, **kw)


def main():
    rows = []; path = HERE / 'results' / 'tuned_all.csv'
    for tick, fn in PANELS.items():
        f = DATA / fn
        if not f.exists():
            continue
        d = load(f); d['q'] = d['date'].dt.to_period('Q')
        qs = sorted(d['q'].unique())
        print(f'\n===== {tick} =====', flush=True)
        for i in range(4, len(qs)):
            tr = d[d['q'].isin(qs[i-4:i])].sort_values('date')
            te = d[d['q'] == qs[i]]
            if len(tr) < 2000 or len(te) < 500:
                continue
            rng = np.random.default_rng(SEED)
            if len(tr) > MAX_TR:
                tr = tr.iloc[np.sort(rng.choice(len(tr), MAX_TR, replace=False))]
            if len(te) > MAX_TE:
                te = te.iloc[rng.choice(len(te), MAX_TE, replace=False)]
            A, Ate = augment(tr[BASE].values), augment(te[BASE].values)
            ytr, yte = tr['IV'].values, te['IV'].values
            dd = A.shape[1]; cut = int(0.8 * len(ytr)); fold = str(qs[i])

            gr = np.random.default_rng(SEED)
            grid = [GRID[k] for k in gr.choice(len(GRID), BUDGET, replace=False)]
            t0 = time.perf_counter(); best = (None, -9e9)
            for prm in grid:
                M = blocks_at(A[:cut], prm['tau'])
                m = et(M / dd, N_IN, min_samples_leaf=prm['leaf'],
                       max_depth=prm['depth'], min_samples_split=prm['split'])
                s = r2_score(ytr[cut:], m.fit(A[:cut], ytr[:cut]).predict(A[cut:]))
                if s > best[1]: best = (prm, s)
            prm = best[0]; search_s = time.perf_counter() - t0

            out = {}
            for name, kw, tau in (('Dynamic ET (tuned)',
                                   dict(min_samples_leaf=prm['leaf'],
                                        max_depth=prm['depth'],
                                        min_samples_split=prm['split']), prm['tau']),
                                  ('Dynamic ET', dict(min_samples_leaf=1), 0.75)):
                M = blocks_at(A, tau)
                t0 = time.perf_counter()
                p = et(M / dd, N_FULL, **kw).fit(A, ytr).predict(Ate)
                out[name] = (r2_score(yte, p), mean_absolute_error(yte, p),
                             float(np.sqrt(mean_squared_error(yte, p))),
                             time.perf_counter() - t0, M)
                rows.append(dict(ticker=tick, fold=fold, model=name,
                                 r2=out[name][0], mae=out[name][1],
                                 rmse=out[name][2], fit_s=out[name][3],
                                 M_blocks=M, d=dd,
                                 search_s=search_s if 'tuned' in name else 0.0,
                                 params=str(prm) if 'tuned' in name else ''))
            print(f'  {fold}  tuned {out["Dynamic ET (tuned)"][0]:.4f}  '
                  f'as specified {out["Dynamic ET"][0]:.4f}  '
                  f'({search_s:.0f}s search)  {prm}', flush=True)
            pd.DataFrame(rows).to_csv(path, index=False)

    t = pd.DataFrame(rows)
    p = t.pivot_table(index=['ticker', 'fold'], columns='model', values='r2')
    d = (p['Dynamic ET (tuned)'] - p['Dynamic ET']).dropna()
    print(f'\n{len(t)} fits over {len(p)} folds -> {path}\n')
    print(t.pivot_table(index='model', columns='ticker', values='r2', aggfunc='mean')
           .assign(all=lambda x: t.groupby('model').r2.mean())
           .to_string(float_format=lambda x: f'{x:.4f}'))
    print(f'\ntuned - as specified: {d.mean():+.4f}, ahead {int((d>0).sum())}/{len(d)}, '
          f'p={wilcoxon(d, alternative="greater").pvalue:.4g}')


if __name__ == '__main__':
    main()
