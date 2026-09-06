"""t18_tuned.py -- every method tuned on the training window, equal budget.

A tuned method against untuned competitors is not a comparison. Every arm here
gets the same search budget on the same temporal inner split of the same training
window, with the same seed, and the test quarter enters nothing. Dynamic ET is
searched over the quantities that belong to it: tau, which sets how many blocks
the window yields and therefore the candidate budget M/d, together with the leaf
and depth controls. The competitors are searched over their own standard grids at
the same number of configurations, forty, drawn at random from each
arm's own grid with a shared seed.

The inner split is temporal, not random: the first four fifths of the calendar
fit, the last fifth scores. A random inner split would leak across the calendar.
The winner is refit on the whole window at full size before the test quarter is
read.

Cost is reported in two parts, search seconds and final fit seconds, so the price
of tuning is visible rather than absorbed.
"""
import sys, time, warnings, itertools
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.ensemble import (ExtraTreesRegressor, RandomForestRegressor,
                              GradientBoostingRegressor)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from t4_clean_sweep import load, augment, BASE

MAX_TR, MAX_TE, SEED = 30_000, 15_000, 42
N_IN, N_FULL, BUDGET = 300, 1000, 40
HERE = Path(__file__).resolve().parents[1]


def blocks_at(X, tau):
    """Union-find over |corr| > tau; returns the number of blocks."""
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


# ---- search spaces, one per arm -------------------------------------------
DYNET_GRID = [dict(tau=t, leaf=l, depth=dp, split=sp)
              for t, l, dp, sp in itertools.product(
                  [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
                  [1, 2, 3, 5], [None, 24], [2, 5])]


def sample(rng, space, k):
    keys = list(space)
    return [{a: space[a][rng.integers(len(space[a]))] for a in keys} for _ in range(k)]


SPACES = {
    'ET': dict(max_features=[0.3, 0.5, 0.7, 0.85, 1.0], min_samples_leaf=[1, 2, 3, 5],
               max_depth=[None, 24], min_samples_split=[2, 5]),
    'RF': dict(max_features=[0.3, 0.5, 0.7, 1.0], min_samples_leaf=[1, 2, 3, 5],
               max_depth=[None, 12, 24], min_samples_split=[2, 5]),
    'XGBoost': dict(max_depth=[4, 6, 8, 10], learning_rate=[0.02, 0.05, 0.1, 0.2],
                    subsample=[0.7, 0.85, 1.0], colsample_bytree=[0.6, 0.8, 1.0],
                    min_child_weight=[1, 5, 20], reg_lambda=[0.5, 1.0, 5.0]),
    'LightGBM': dict(num_leaves=[31, 63, 127, 255], learning_rate=[0.02, 0.05, 0.1, 0.2],
                     subsample=[0.7, 0.85, 1.0], colsample_bytree=[0.6, 0.8, 1.0],
                     min_child_samples=[5, 20, 50], reg_lambda=[0.0, 1.0, 5.0]),
    'CatBoost': dict(depth=[4, 6, 8, 10], learning_rate=[0.02, 0.05, 0.1, 0.2],
                     l2_leaf_reg=[1.0, 3.0, 10.0], subsample=[0.7, 0.85, 1.0]),
}


def build(name, prm, n_trees):
    if name == 'ET':
        return ExtraTreesRegressor(n_estimators=n_trees, bootstrap=False, n_jobs=-1,
                                   random_state=SEED, **prm)
    if name == 'RF':
        return RandomForestRegressor(n_estimators=min(n_trees, 300), n_jobs=-1,
                                     random_state=SEED, **prm)
    if name == 'XGBoost':
        return XGBRegressor(n_estimators=600, n_jobs=-1, random_state=SEED,
                            verbosity=0, **prm)
    if name == 'LightGBM':
        return LGBMRegressor(n_estimators=600, n_jobs=-1, random_state=SEED,
                             verbose=-1, **prm)
    if name == 'CatBoost':
        return CatBoostRegressor(iterations=600, random_seed=SEED, verbose=0,
                                 thread_count=-1, **prm)
    if name == 'GBR':
        return GradientBoostingRegressor(n_estimators=300, random_state=SEED, **prm)
    raise KeyError(name)


def main():
    d = load(); d['q'] = d['date'].dt.to_period('Q')
    qs = sorted(d['q'].unique()); rows = []
    path = HERE / 'results' / 'tuned.csv'

    for i in range(4, len(qs)):
        tr = d[d['q'].isin(qs[i-4:i])].sort_values('date')
        te = d[d['q'] == qs[i]]
        if len(tr) < 500 or len(te) < 50:
            continue
        rng = np.random.default_rng(SEED)
        if len(tr) > MAX_TR:
            tr = tr.iloc[np.sort(rng.choice(len(tr), MAX_TR, replace=False))]
        if len(te) > MAX_TE:
            te = te.iloc[rng.choice(len(te), MAX_TE, replace=False)]
        B_tr, B_te = tr[BASE].values, te[BASE].values
        A_tr, A_te = augment(B_tr), augment(B_te)
        ytr, yte = tr['IV'].values, te['IV'].values
        dd = A_tr.shape[1]
        cut = int(0.8 * len(ytr))
        fold = str(qs[i])
        print(f'\nFold {fold}  train={len(tr):,}  test={len(te):,}', flush=True)

        def record(name, prm, search_s, mdl, X, Xte, dcol):
            t0 = time.perf_counter(); mdl.fit(X, ytr); fit_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            p = np.asarray(mdl.predict(Xte), dtype=float)
            pred_s = time.perf_counter() - t0
            rec = dict(fold=fold, model=name, d=dcol, params=str(prm),
                       r2=r2_score(yte, p), mae=mean_absolute_error(yte, p),
                       rmse=float(np.sqrt(mean_squared_error(yte, p))),
                       search_s=search_s, fit_s=fit_s, pred_s=pred_s)
            rows.append(rec)
            print(f'  {name:<20} R2={rec["r2"]:>7.4f}  MAE={rec["mae"]:.4f}  '
                  f'search {search_s:6.1f}s  fit {fit_s:6.1f}s  {prm}', flush=True)
            pd.DataFrame(rows).to_csv(path, index=False)

        # ---- Dynamic ET, searched over tau and the tree controls -----------
        gr = np.random.default_rng(SEED)
        grid = [DYNET_GRID[k] for k in
                gr.choice(len(DYNET_GRID), BUDGET, replace=False)]
        t0 = time.perf_counter(); best = (None, -9e9)
        for prm in grid:
            M = blocks_at(A_tr[:cut], prm['tau'])
            m = ExtraTreesRegressor(n_estimators=N_IN, max_features=M / dd,
                                    min_samples_leaf=prm['leaf'],
                                    max_depth=prm['depth'],
                                    min_samples_split=prm['split'],
                                    bootstrap=False, n_jobs=-1, random_state=SEED)
            s = r2_score(ytr[cut:], m.fit(A_tr[:cut], ytr[:cut]).predict(A_tr[cut:]))
            if s > best[1]: best = (prm, s)
        prm = best[0]; search_s = time.perf_counter() - t0
        M = blocks_at(A_tr, prm['tau'])
        record('Dynamic ET (tuned)', dict(prm, M=M, mf=round(M / dd, 3)), search_s,
               ExtraTreesRegressor(n_estimators=N_FULL, max_features=M / dd,
                                   min_samples_leaf=prm['leaf'],
                                   max_depth=prm['depth'],
                                   min_samples_split=prm['split'],
                                   bootstrap=False, n_jobs=-1, random_state=SEED),
               A_tr, A_te, dd)

        # ---- the untuned form the manuscript ships -------------------------
        M0 = blocks_at(A_tr, 0.75)
        record('Dynamic ET', dict(tau=0.75, M=M0, mf=round(M0 / dd, 3)), 0.0,
               ExtraTreesRegressor(n_estimators=N_FULL, max_features=M0 / dd,
                                   min_samples_leaf=1, bootstrap=False,
                                   n_jobs=-1, random_state=SEED), A_tr, A_te, dd)

        # ---- ET-aug: the ablation, all candidates on the same features -----
        record('ET-aug', dict(max_features=1.0), 0.0,
               ExtraTreesRegressor(n_estimators=N_FULL, max_features=1.0,
                                   min_samples_leaf=1, bootstrap=False,
                                   n_jobs=-1, random_state=SEED), A_tr, A_te, dd)

        # ---- competitors, equal budget on the base features ----------------
        for name, space in SPACES.items():
            r2 = np.random.default_rng(SEED)
            t0 = time.perf_counter(); best = (None, -9e9)
            for prm in sample(r2, space, BUDGET):
                try:
                    m = build(name, prm, N_IN)
                    s = r2_score(ytr[cut:],
                                 m.fit(B_tr[:cut], ytr[:cut]).predict(B_tr[cut:]))
                except Exception:
                    continue
                if s > best[1]: best = (prm, s)
            prm = best[0]; search_s = time.perf_counter() - t0
            if prm is None:
                continue
            record(f'{name} (tuned)', prm, search_s, build(name, prm, N_FULL),
                   B_tr, B_te, B_tr.shape[1])

    t = pd.DataFrame(rows)
    g = (t.groupby('model')[['r2', 'mae', 'search_s', 'fit_s', 'pred_s']].mean()
          .sort_values('r2', ascending=False))
    print(f'\n{len(t)} fits over {t.fold.nunique()} folds -> {path}\n')
    print(g.to_string(float_format=lambda x: f'{x:.4f}'))


if __name__ == '__main__':
    main()
