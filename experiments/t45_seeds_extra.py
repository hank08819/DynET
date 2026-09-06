"""t45_seeds_extra.py -- 图 12 的另外四条曲线, 与 t42_seeds 同一协议.

t42_seeds 已经在五个随机状态下跑完了三条以 Extra Trees 为核心的曲线, 结果在
seeds.csv. 这里补上论文表中出现的另外四个对照: CatBoost, XGBoost, 随机森林和
不做增广的 Extra Trees. 它们都在七列原始特征上训练, 与第 VI 节的表保持一致.

单独成文而不并入 t42_seeds, 是因为 Block Forests 的实现依赖 numba 的并行域,
与三个梯度提升库各自带来的 libomp 放在同一个进程里会互相锁死. 这个脚本不引入
numba, 跑完之后与 seeds.csv 合并成 seeds7.csv.
"""
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from t4_clean_sweep import load, BASE

MAX_TR, MAX_TE, DATA_SEED, NT = 30_000, 15_000, 42, 1000
SEEDS = [42, 7, 101, 2024, 13]
HERE = Path(__file__).resolve().parents[1]


def main():
    d = load(); d['q'] = d['date'].dt.to_period('Q')
    qs = sorted(d['q'].unique()); rows = []
    path = HERE / 'results' / 'seeds_extra.csv'

    for i in range(4, len(qs)):
        tr = d[d['q'].isin(qs[i-4:i])]; te = d[d['q'] == qs[i]]
        if len(tr) < 500 or len(te) < 50: continue
        rng = np.random.default_rng(DATA_SEED)
        if len(tr) > MAX_TR: tr = tr.iloc[rng.choice(len(tr), MAX_TR, replace=False)]
        if len(te) > MAX_TE: te = te.iloc[rng.choice(len(te), MAX_TE, replace=False)]
        Xtr, Xte = tr[BASE].values, te[BASE].values
        ytr, yte = tr['IV'].values, te['IV'].values
        fold = str(qs[i])

        for s in SEEDS:
            arms = {
                'CatBoost': lambda: CatBoostRegressor(iterations=600, depth=6,
                    learning_rate=0.05, random_seed=s, verbose=0,
                    thread_count=-1).fit(Xtr, ytr).predict(Xte),
                'XGBoost': lambda: XGBRegressor(n_estimators=600, max_depth=6,
                    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                    n_jobs=-1, random_state=s, verbosity=0).fit(Xtr, ytr).predict(Xte),
                'RF': lambda: RandomForestRegressor(n_estimators=300, max_depth=12,
                    min_samples_leaf=3, n_jobs=-1, random_state=s
                    ).fit(Xtr, ytr).predict(Xte),
                'ET': lambda: ExtraTreesRegressor(n_estimators=NT, max_depth=None,
                    min_samples_leaf=1, max_features=1.0, bootstrap=False,
                    n_jobs=-1, random_state=s).fit(Xtr, ytr).predict(Xte),
            }
            line = []
            for name, fn in arms.items():
                t0 = time.perf_counter()
                p = np.asarray(fn(), dtype=float)
                rows.append(dict(fold=fold, seed=s, model=name, d=Xtr.shape[1],
                                 r2=r2_score(yte, p), mae=mean_absolute_error(yte, p),
                                 secs=time.perf_counter()-t0))
                line.append(f'{name} {rows[-1]["r2"]:.4f}')
            print(f'{fold} seed={s:<5} ' + '  '.join(line), flush=True)
            pd.DataFrame(rows).to_csv(path, index=False)

    t = pd.DataFrame(rows)
    print(f'\n{len(t)} fits over {t.fold.nunique()} folds x {t.seed.nunique()} seeds'
          f' -> {path}\n')
    print(t.pivot_table(index='model', columns='seed', values='r2').round(4).to_string())

    old = pd.read_csv(HERE / 'results' / 'seeds.csv').dropna(subset=['r2'])
    both = pd.concat([old[['fold', 'seed', 'model', 'r2', 'mae', 'secs']],
                      t[['fold', 'seed', 'model', 'r2', 'mae', 'secs']]],
                     ignore_index=True)
    out = HERE / 'results' / 'seeds7.csv'
    both.to_csv(out, index=False)
    print('\nwrote', out, len(both), 'rows,', both.model.nunique(), 'models')


if __name__ == '__main__':
    main()
