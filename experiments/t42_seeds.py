"""t42_seeds.py -- the two load-bearing margins across five random states.

Extra Trees is a randomised estimator and the margins this paper reports are in
the third decimal, so the ranking has to be shown to survive the draw. The row
subsample stays at the protocol's seed of 42 at every repetition; only the
ensemble's random_state moves. That isolates the quantity in question, which is
the randomisation of the forest rather than the composition of the window.

Three arms, the ones the paper's claims rest on:

  Dynamic ET   the rule as specified, max_features = M/d
  ET-aug       the same fifteen columns, all candidates at every node
  BF-default   Block Forests in its most commonly used setting, on the a-priori
               semantic grouping, which is the head-to-head of Section 7

The import is from t4_clean_sweep for the reason recorded in t37: t24 loads
three libraries that each bring their own libomp.
"""
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from t4_clean_sweep import load, augment, blocks, BASE
from blocked_et import BlockedExtraTrees

MAX_TR, MAX_TE, DATA_SEED, NT = 30_000, 15_000, 42, 1000
SEEDS = [42, 7, 101, 2024, 13]
SEM = np.array([0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2])
HERE = Path(__file__).resolve().parents[1]


def main():
    d = load(); d['q'] = d['date'].dt.to_period('Q')
    qs = sorted(d['q'].unique()); rows = []
    path = HERE / 'results' / 'seeds.csv'
    sem = np.unique(SEM, return_inverse=True)[1]
    mv_sem = np.array([max(1, int(np.sqrt((sem == b).sum())))
                       for b in range(sem.max() + 1)])

    for i in range(4, len(qs)):
        tr = d[d['q'].isin(qs[i-4:i])]; te = d[d['q'] == qs[i]]
        if len(tr) < 500 or len(te) < 50: continue
        rng = np.random.default_rng(DATA_SEED)
        if len(tr) > MAX_TR: tr = tr.iloc[rng.choice(len(tr), MAX_TR, replace=False)]
        if len(te) > MAX_TE: te = te.iloc[rng.choice(len(te), MAX_TE, replace=False)]
        Xtr, Xte = augment(tr[BASE].values), augment(te[BASE].values)
        ytr, yte = tr['IV'].values, te['IV'].values
        dd = Xtr.shape[1]; M = blocks(Xtr); fold = str(qs[i])

        for s in SEEDS:
            arms = {
                'Dynamic ET': lambda: ExtraTreesRegressor(
                    n_estimators=NT, max_depth=None, min_samples_leaf=1,
                    max_features=M / dd, bootstrap=False, n_jobs=-1,
                    random_state=s).fit(Xtr, ytr).predict(Xte),
                'ET-aug': lambda: ExtraTreesRegressor(
                    n_estimators=NT, max_depth=None, min_samples_leaf=1,
                    max_features=1.0, bootstrap=False, n_jobs=-1,
                    random_state=s).fit(Xtr, ytr).predict(Xte),
                'BF-default': lambda: BlockedExtraTrees(
                    NT, mode=4, mvec=mv_sem, blocks=sem,
                    random_state=s).fit(Xtr, ytr).predict(Xte),
            }
            line = []
            for name, fn in arms.items():
                t0 = time.perf_counter()
                p = np.asarray(fn(), dtype=float)
                rows.append(dict(fold=fold, seed=s, model=name, M_blocks=M, d=dd,
                                 r2=r2_score(yte, p), mae=mean_absolute_error(yte, p),
                                 secs=time.perf_counter()-t0))
                line.append(f'{name} {rows[-1]["r2"]:.4f}')
            print(f'{fold} seed={s:<5} ' + '  '.join(line), flush=True)
            pd.DataFrame(rows).to_csv(path, index=False)

    t = pd.DataFrame(rows)
    print(f'\n{len(t)} fits over {t.fold.nunique()} folds x {t.seed.nunique()} seeds'
          f' -> {path}\n')
    print(t.pivot_table(index='model', columns='seed', values='r2').round(4).to_string())


if __name__ == '__main__':
    main()
