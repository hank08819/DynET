"""t23b_timing_torch.py -- cost of the neural arms, separate process.

The boosting libraries load their own OpenMP runtime and torch aborts under
it, so the neural arms are timed in a process that never imports them. Same
machine, same folds, same rows.

Accuracy comes from the sixteen-fold runs. Cost has to be measured where nothing
else is competing for the cores, so it is measured here: three representative
folds, every arm, the same rows in the same order, one process on the machine.
The seconds are the mean per fold to fit 30,000 rows and to predict 15,000.
"""
import sys, time, warnings, platform, os
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from t4_clean_sweep import load, augment, blocks, BASE
from seq_baseline import SeqRegressor
from itransformer import ITransformerRegressor
from ft_transformer import FTTransformerRegressor
from dynetmer import DynETmer
from dynetmer2 import DynETmer2

MAX_TR, MAX_TE, SEED, NT = 30_000, 15_000, 42, 1000
FOLDS = [6, 10, 14]                       # three spread across the panel
HERE = Path(__file__).resolve().parents[1]


def main():
    d = load(); d['q'] = d['date'].dt.to_period('Q')
    qs = sorted(d['q'].unique()); rows = []
    path = HERE / 'results' / 'timing_torch.csv'
    print(f'{platform.platform()}  cores={os.cpu_count()}', flush=True)

    for i in FOLDS:
        tr = d[d['q'].isin(qs[i-4:i])]; te = d[d['q'] == qs[i]]
        rng = np.random.default_rng(SEED)
        if len(tr) > MAX_TR: tr = tr.iloc[rng.choice(len(tr), MAX_TR, replace=False)]
        if len(te) > MAX_TE: te = te.iloc[rng.choice(len(te), MAX_TE, replace=False)]
        Btr, Bte = tr[BASE].values, te[BASE].values
        Atr, Ate = augment(Btr), augment(Bte)
        ytr, yte = tr['IV'].values, te['IV'].values
        M, dd = blocks(Atr), Atr.shape[1]
        print(f'\nFold {qs[i]}  M={M}/{dd}', flush=True)

        def et(mf):
            return ExtraTreesRegressor(n_estimators=NT, max_depth=None,
                                       min_samples_leaf=1, max_features=mf,
                                       bootstrap=False, n_jobs=-1, random_state=SEED)
        arms = [
            ('iTransformer',   Btr, Bte, lambda: ITransformerRegressor(seed=SEED)),
            ('FT-Transformer', Btr, Bte, lambda: FTTransformerRegressor(seed=SEED)),
            ('DynET-mer',      Btr, Bte, lambda: DynETmer(
                seed=SEED, augment=augment, whiten='none')),
            ('DynET-mer2',     Btr, Bte, lambda: DynETmer2(
                encoder='it', seed=SEED, augment=augment)),
            ('LSTM',           Btr, Bte, lambda: SeqRegressor(kind='lstm', seed=SEED)),
            ('GRU',            Btr, Bte, lambda: SeqRegressor(kind='gru', seed=SEED)),
        ]
        for name, X, Xte, make in arms:
            try:
                m = make()
                t0 = time.perf_counter(); m.fit(X, ytr); fit_s = time.perf_counter() - t0
                t0 = time.perf_counter()
                p = np.asarray(m.predict(Xte), dtype=float)
                pred_s = time.perf_counter() - t0
                rows.append(dict(fold=str(qs[i]), model=name, fit_s=fit_s,
                                 pred_s=pred_s, r2=r2_score(yte, p)))
                print(f'  {name:<16} fit {fit_s:7.2f}s  pred {pred_s:5.2f}s  '
                      f'R2={rows[-1]["r2"]:.4f}', flush=True)
            except Exception as e:
                print(f'  {name:<16} FAILED: {str(e)[:80]}', flush=True)
            pd.DataFrame(rows).to_csv(path, index=False)

    t = pd.DataFrame(rows)
    g = t.groupby('model')[['fit_s', 'pred_s']].mean().sort_values('fit_s')
    print(f'\n{len(t)} fits over {t.fold.nunique()} folds -> {path}\n')
    print(g.to_string(float_format=lambda x: f'{x:.2f}'))


if __name__ == '__main__':
    main()
