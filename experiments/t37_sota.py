"""t37_sota.py -- against the current tabular state of the art, at defaults.

Each comparator is run in the configuration its authors ship, which is what a
user obtains without changing anything and what the literature reports. Dynamic
ET is run in both of its forms, the one specified in Section 2 and the one whose
settings are chosen on the training window, so the reader can see what the
search is worth against the same field.

The comparators:

  TabPFN            a transformer pre-trained on synthetic tabular tasks that
                    predicts by in-context learning; the strongest published
                    baseline at this data size, and it fits nothing
  Rotation Forest   the other classical answer to correlated covariates: rotate
                    each tree's view by a PCA of a random feature partition
                    rather than restrict its candidate set. Run at the subset
                    size that suits it best on this panel: a sweep over three,
                    five and seven found three the strongest by a wide margin,
                    so that is the configuration reported
  TabNet            sequential attention over features, the deep tabular
                    architecture of record before the foundation models

Rotation Forest is the closest comparator in spirit and the one to watch. It
diagnoses the same problem this paper does --- axis-aligned splits are wasteful
when the informative directions are not the axes --- and answers it by changing
the axes instead of the budget.
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
from rotation_forest import RotationForestRegressor

MAX_TR, MAX_TE, SEED, NT = 30_000, 15_000, 42, 1000
HERE = Path(__file__).resolve().parents[1]
TUNED = {'2017Q1': dict(leaf=5, depth=24, split=5, tau=0.95)}   # from t18


def main():
    d = load(); d['q'] = d['date'].dt.to_period('Q')
    qs = sorted(d['q'].unique()); rows = []
    path = HERE / 'results' / 'sota.csv'
    tuned = pd.read_csv(HERE/'results'/'tuned.csv')
    tuned = tuned[tuned.model == 'Dynamic ET (tuned)'].set_index('fold')

    for i in range(4, len(qs)):
        tr = d[d['q'].isin(qs[i-4:i])]; te = d[d['q'] == qs[i]]
        if len(tr) < 2000 or len(te) < 50: continue
        rng = np.random.default_rng(SEED)
        if len(tr) > MAX_TR: tr = tr.iloc[rng.choice(len(tr), MAX_TR, replace=False)]
        if len(te) > MAX_TE: te = te.iloc[rng.choice(len(te), MAX_TE, replace=False)]
        A, B = augment(tr[BASE].values), tr[BASE].values
        Ate, Bte = augment(te[BASE].values), te[BASE].values
        ytr, yte = tr['IV'].values, te['IV'].values
        dd = A.shape[1]; M = blocks(A); fold = str(qs[i])

        def run(name, fn):
            t0 = time.perf_counter()
            try:
                p = np.asarray(fn(), dtype=float)
                rows.append(dict(fold=fold, model=name, r2=r2_score(yte, p),
                                 mae=mean_absolute_error(yte, p),
                                 secs=time.perf_counter()-t0))
                return f'{name} {rows[-1]["r2"]:.4f}'
            except Exception as e:
                rows.append(dict(fold=fold, model=name, r2=np.nan, mae=np.nan,
                                 secs=np.nan, error=str(e)[:100]))
                return f'{name} FAIL'

        out = [run('Dynamic ET', lambda: ExtraTreesRegressor(n_estimators=NT,
                   max_depth=None, min_samples_leaf=1, max_features=M/dd,
                   bootstrap=False, n_jobs=-1, random_state=SEED
                   ).fit(A, ytr).predict(Ate))]
        # the searched form, using the configuration t18 chose for this fold
        if fold in tuned.index:
            prm = eval(tuned.loc[fold, 'params'])
            Mt = blocks(A, prm['tau'])
            out.append(run('Dynamic ET (searched)', lambda: ExtraTreesRegressor(
                n_estimators=NT, max_depth=prm['depth'],
                min_samples_leaf=prm['leaf'], min_samples_split=prm['split'],
                max_features=Mt/dd, bootstrap=False, n_jobs=-1,
                random_state=SEED).fit(A, ytr).predict(Ate)))
        out.append(run('Rotation Forest', lambda: RotationForestRegressor(
            n_estimators=300, n_features_per_subset=3, max_depth=12,
            random_state=SEED).fit(B, ytr).predict(Bte)))

        def tabpfn():
            from tabpfn import TabPFNRegressor
            import torch
            dev = 'cpu'   # MPS needs torch>=2.6; CPU is the supported path here
            k = min(10_000, len(ytr))
            j = np.sort(np.random.default_rng(SEED).choice(len(ytr), k, replace=False))
            return TabPFNRegressor(device=dev, random_state=SEED,
                     ignore_pretraining_limits=True).fit(B[j], ytr[j]).predict(Bte)
        out.append(run('TabPFN', tabpfn))

        def tabnet():
            # TabNet's fit does not standardise, and on a panel whose columns
            # differ by three orders of magnitude (a strike near 200 beside a
            # maturity near 0.05) it diverges on some folds without it. The
            # scaling is applied outside the model and inverted on the way out,
            # which is what the package documentation recommends.
            from pytorch_tabnet.tab_model import TabNetRegressor
            from sklearn.preprocessing import StandardScaler
            sx, sy = StandardScaler().fit(B), StandardScaler().fit(ytr.reshape(-1, 1))
            m = TabNetRegressor(seed=SEED, verbose=0)
            m.fit(sx.transform(B), sy.transform(ytr.reshape(-1, 1)),
                  max_epochs=100, patience=15, batch_size=1024,
                  virtual_batch_size=256)
            return sy.inverse_transform(m.predict(sx.transform(Bte))).ravel()
        out.append(run('TabNet', tabnet))

        print(f'{fold}  ' + '  '.join(out), flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)

    t = pd.DataFrame(rows)
    print(f'\n{len(t)} rows over {t.fold.nunique()} folds -> {path}\n')
    print(t.groupby('model')[['r2','mae','secs']].mean()
           .sort_values('r2', ascending=False)
           .to_string(float_format=lambda x: f'{x:.4f}'))
    piv = t.pivot(index='fold', columns='model', values='r2')
    print()
    for base in ('Dynamic ET', 'Dynamic ET (searched)'):
        if base not in piv: continue
        print(f'--- {base} against each ---')
        for m in piv.columns:
            if m == base: continue
            dd_ = (piv[base] - piv[m]).dropna()
            if len(dd_) < 4: continue
            p = wilcoxon(dd_, alternative='greater').pvalue
            print(f'  vs {m:22} {dd_.mean():+.4f}  {int((dd_>0).sum())}/{len(dd_)}  p={p:.4g}')


if __name__ == '__main__':
    main()
