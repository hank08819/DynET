"""t46_seeds_more.py -- 图 12 再补两条: LightGBM 和 Rotation Forest.

与 t42_seeds, t45_seeds_extra 同一协议: 同样的十六个季度窗口, 同样的抽样种子,
只让模型自己的随机状态在五个值之间移动. 两个模型都在七列原始特征上训练, 与论文
第 VI 节的表一致.

不引入 torch, 也不引入 numba, 避免与 LightGBM 自带的 libomp 冲突.
"""
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from lightgbm import LGBMRegressor
from t4_clean_sweep import load, BASE
from rotation_forest import RotationForestRegressor

MAX_TR, MAX_TE, DATA_SEED = 30_000, 15_000, 42
SEEDS = [42, 7, 101, 2024, 13]
HERE = Path(__file__).resolve().parents[1]


def main():
    d = load(); d['q'] = d['date'].dt.to_period('Q')
    qs = sorted(d['q'].unique()); rows = []
    path = HERE / 'results' / 'seeds_more.csv'

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
                'LightGBM': lambda: LGBMRegressor(n_estimators=600, max_depth=6,
                    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                    n_jobs=-1, random_state=s, verbose=-1
                    ).fit(Xtr, ytr).predict(Xte),
                'Rotation Forest': lambda: RotationForestRegressor(
                    n_estimators=300, n_features_per_subset=3, max_depth=12,
                    random_state=s).fit(Xtr, ytr).predict(Xte),
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


if __name__ == '__main__':
    main()
