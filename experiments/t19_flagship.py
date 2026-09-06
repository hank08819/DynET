"""t19_flagship.py -- Dynamic ET tuned on the levers that move the number.

t18 searches the tree controls. Those move the third decimal. The quantities that
move the second are the ones the block rule reads: which coordinates the surface
is written in, what scale the target is on, and how the training window is
weighted across the calendar. This searches those, on the training window only.

  augmentation  the eight interaction terms the manuscript ships, and richer sets
                that add the standardised log-moneyness lm/sqrt(T) and the total
                variance proxy lm^2/T -- the coordinates a volatility surface is
                naturally parameterised in, which the shipped set does not carry
  target        implied volatility is positive and right-skewed; log IV is the
                scale on which its errors are closer to homoscedastic
  recency       a walk-forward window spans four quarters and the near quarter is
                the more informative one; an exponential half-life says how much
  leaf          carried through from t18's range

Every choice is made on a temporal inner split of the training window and the
test quarter enters nothing. Blocks are inferred on whatever augmented matrix the
search selects, so the candidate budget M/d adapts to the chosen coordinates.
"""
import sys, time, warnings, itertools
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import wilcoxon
from t4_clean_sweep import load, augment, BASE

MAX_TR, MAX_TE, SEED = 30_000, 15_000, 42
N_IN, N_FULL = 300, 1000
HERE = Path(__file__).resolve().parents[1]


def aug0(X):
    """The eight terms the manuscript ships."""
    return augment(X)


def aug1(X):
    """Adds the surface coordinates: standardised log-moneyness and total variance."""
    m, lm, t, sp = X[:, 0], X[:, 1], X[:, 2], X[:, 6]
    st = np.sqrt(np.where(t > 1e-8, t, 1e-8)); tt = np.where(t > 1e-8, t, 1e-8)
    return np.hstack([aug0(X), np.column_stack([lm / st, lm**2 / tt, st, lm / tt])])


def aug2(X):
    """Adds the call/put and relative-spread interactions on top of aug1."""
    m, lm, t, S, c, sp = X[:, 0], X[:, 1], X[:, 2], X[:, 4], X[:, 5], X[:, 6]
    st = np.sqrt(np.where(t > 1e-8, t, 1e-8))
    Ss = np.where(np.abs(S) > 1e-8, S, 1e-8)
    return np.hstack([aug1(X), np.column_stack([lm * c, m * c, sp / Ss,
                                                (lm / st) * c])])


AUGS = {'ship8': aug0, 'surface12': aug1, 'surface16': aug2}
TARGETS = ['raw', 'log']
HALFLIFE = [None, 120.0, 250.0]        # days
LEAF = [1, 2]


def blocks_at(X, tau=0.75):
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


def weights(dates, hl):
    if hl is None:
        return None
    age = (dates.max() - dates).astype('timedelta64[D]').astype(float)
    return np.power(0.5, age / hl)


def fit_predict(Xtr, ytr, Xte, *, target, leaf, mf, w, n_trees):
    yy = np.log(np.clip(ytr, 1e-6, None)) if target == 'log' else ytr
    m = ExtraTreesRegressor(n_estimators=n_trees, max_depth=None,
                            min_samples_leaf=leaf, max_features=mf,
                            bootstrap=False, n_jobs=-1, random_state=SEED)
    m.fit(Xtr, yy, sample_weight=w)
    p = m.predict(Xte)
    return np.exp(p) if target == 'log' else p


def main():
    d = load(); d['q'] = d['date'].dt.to_period('Q')
    qs = sorted(d['q'].unique()); rows = []
    path = HERE / 'results' / 'flagship.csv'

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
        Btr, Bte = tr[BASE].values, te[BASE].values
        ytr, yte = tr['IV'].values, te['IV'].values
        dtr = tr['date'].values
        cut = int(0.8 * len(ytr))
        fold = str(qs[i])
        print(f'\nFold {fold}  train={len(tr):,}  test={len(te):,}', flush=True)

        # ---- search on the training window only ---------------------------
        t0 = time.perf_counter(); best = (None, -9e9)
        for an, af in AUGS.items():
            A = af(Btr); dd = A.shape[1]
            M = blocks_at(A[:cut]); mf = M / dd
            for tgt, hl, leaf in itertools.product(TARGETS, HALFLIFE, LEAF):
                w = weights(dtr[:cut], hl)
                p = fit_predict(A[:cut], ytr[:cut], A[cut:], target=tgt, leaf=leaf,
                                mf=mf, w=w, n_trees=N_IN)
                s = r2_score(ytr[cut:], p)
                if s > best[1]:
                    best = (dict(aug=an, target=tgt, halflife=hl, leaf=leaf), s)
        cfg, inner = best; search_s = time.perf_counter() - t0

        # ---- refit the winner on the whole window --------------------------
        A, Ate = AUGS[cfg['aug']](Btr), AUGS[cfg['aug']](Bte)
        dd = A.shape[1]; M = blocks_at(A); mf = M / dd
        w = weights(dtr, cfg['halflife'])
        t0 = time.perf_counter()
        p = fit_predict(A, ytr, Ate, target=cfg['target'], leaf=cfg['leaf'],
                        mf=mf, w=w, n_trees=N_FULL)
        fit_s = time.perf_counter() - t0

        # ---- the shipped form, same fold, for the paired test --------------
        A0, A0te = aug0(Btr), aug0(Bte)
        M0 = blocks_at(A0)
        t0 = time.perf_counter()
        p0 = fit_predict(A0, ytr, A0te, target='raw', leaf=1,
                         mf=M0 / A0.shape[1], w=None, n_trees=N_FULL)
        fit0_s = time.perf_counter() - t0

        rec = dict(fold=fold, **cfg, M=M, d=dd, mf=round(mf, 3), inner_r2=inner,
                   r2=r2_score(yte, p), mae=mean_absolute_error(yte, p),
                   rmse=float(np.sqrt(mean_squared_error(yte, p))),
                   base_r2=r2_score(yte, p0), base_mae=mean_absolute_error(yte, p0),
                   search_s=search_s, fit_s=fit_s, base_fit_s=fit0_s)
        rows.append(rec)
        print(f"  chose {cfg['aug']} target={cfg['target']} hl={cfg['halflife']} "
              f"leaf={cfg['leaf']}  M={M}/{dd}", flush=True)
        print(f"  tuned R2={rec['r2']:.4f}  shipped R2={rec['base_r2']:.4f}  "
              f"delta={rec['r2']-rec['base_r2']:+.4f}   "
              f"search {search_s:.0f}s  fit {fit_s:.1f}s", flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)

    t = pd.DataFrame(rows)
    dlt = t.r2 - t.base_r2
    p = wilcoxon(dlt, alternative='greater').pvalue if len(dlt) > 2 else np.nan
    print(f'\n{len(t)} folds -> {path}')
    print(f'tuned   mean R2 = {t.r2.mean():.4f}   MAE = {t.mae.mean():.4f}   '
          f'fit {t.fit_s.mean():.1f}s  (search {t.search_s.mean():.0f}s)')
    print(f'shipped mean R2 = {t.base_r2.mean():.4f}   MAE = {t.base_mae.mean():.4f} '
          f'  fit {t.base_fit_s.mean():.1f}s')
    print(f'delta = {dlt.mean():+.4f}   ahead on {int((dlt>0).sum())}/{len(dlt)} '
          f'folds   Wilcoxon p={p:.4g}')
    for c in ('aug', 'target', 'halflife', 'leaf'):
        print(f'  chosen {c}: {t[c].value_counts().to_dict()}')


if __name__ == '__main__':
    main()
