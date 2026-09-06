"""t23a_timing_cpu.py -- cost of the CPU arms, measured on an idle machine.

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
from sklearn.ensemble import (ExtraTreesRegressor, RandomForestRegressor,
                              GradientBoostingRegressor)
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from t4_clean_sweep import load, augment, blocks, BASE
from svi_baseline import SVIRegressor
from heston_baseline import HestonRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MAX_TR, MAX_TE, SEED, NT = 30_000, 15_000, 42, 1000
FOLDS = [6, 10, 14]                       # three spread across the panel
HERE = Path(__file__).resolve().parents[1]


def main():
    d = load(); d['q'] = d['date'].dt.to_period('Q')
    qs = sorted(d['q'].unique()); rows = []
    path = HERE / 'results' / 'timing_cpu.csv'
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
            ('Dynamic ET',       Atr, Ate, lambda: et(M / dd)),
            ('ET-aug',           Atr, Ate, lambda: et(1.0)),
            ('ET',               Btr, Bte, lambda: et(1.0)),
            ('RF',               Btr, Bte, lambda: RandomForestRegressor(
                n_estimators=300, max_depth=12, min_samples_leaf=3,
                n_jobs=-1, random_state=SEED)),
            ('GBR',              Btr, Bte, lambda: GradientBoostingRegressor(
                n_estimators=200, max_depth=4, random_state=SEED)),
            ('XGBoost',          Btr, Bte, lambda: XGBRegressor(
                n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, n_jobs=-1, random_state=SEED, verbosity=0)),
            ('LightGBM',         Btr, Bte, lambda: LGBMRegressor(
                n_estimators=600, num_leaves=63, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, n_jobs=-1, random_state=SEED, verbose=-1)),
            ('CatBoost',         Btr, Bte, lambda: CatBoostRegressor(
                iterations=600, depth=6, learning_rate=0.05, random_seed=SEED,
                verbose=0, thread_count=-1)),
            ('SVI',              Btr, Bte, lambda: SVIRegressor()),
            ('Heston',           Btr, Bte, lambda: HestonRegressor()),
            ('MLP',              Btr, Bte, lambda: make_pipeline(
                StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 64),
                max_iter=400, early_stopping=True, random_state=SEED))),
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
