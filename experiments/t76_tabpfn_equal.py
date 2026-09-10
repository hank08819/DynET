"""t76_tabpfn_equal.py -- TabPFN 在与本方法相同的训练行数下的成绩.

这一支刻意不导入 scikit-learn: sklearn 与 PyTorch 各自带一份 OpenMP 运行时,
同处一进程会段错误(退出码 139), 这与 t63 按库分进程的理由相同. 折的切分、抽样
与上限都与 t75 完全一致, 所以两边读到的是同一批行, 结果可以直接配对.

R^2 直接按定义算, 不经过 sklearn.
"""
import os, sys, time, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np, pandas as pd

CAP = int(os.environ.get('TRAIN_CAP', '10000'))
MAX_TR, MAX_TE, SEED = 30_000, 15_000, 42
BASE = ['m', 'logm', 'T', 'K', 'S', 'is_call', 'spr']
DATA = Path(os.environ.get('OPTDATA',
            '/Users/henry_han/LHP/JMP1+MS2027/data/OptionData2026'))
PANELS = {'AAPL': 'aapl_2016_2020', 'NVDA': 'nvda_2020_2022',
          'QQQ': 'qqq_2020_2022', 'SPY': 'spy_2020_2022', 'TSLA': 'tsla_2019_2022'}
HERE = Path(__file__).resolve().parents[1]


def r2(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main(tick='AAPL'):
    from tabpfn import TabPFNRegressor
    d = pd.read_parquet(DATA / (PANELS[tick] + '.parquet'))
    d['q'] = d['date'].dt.to_period('Q')
    qs = sorted(d['q'].unique()); rows = []
    path = HERE / 'results' / f'equal_n_{tick}_{CAP}_tabpfn.csv'
    for i in range(4, len(qs)):
        tr = d[d['q'].isin(qs[i-4:i])]; te = d[d['q'] == qs[i]]
        if len(tr) < 500 or len(te) < 50: continue
        rng = np.random.default_rng(SEED)
        if len(tr) > MAX_TR: tr = tr.iloc[rng.choice(len(tr), MAX_TR, replace=False)]
        if len(te) > MAX_TE: te = te.iloc[rng.choice(len(te), MAX_TE, replace=False)]
        if len(tr) > CAP:
            j = np.sort(np.random.default_rng(SEED).choice(len(tr), CAP, replace=False))
            tr = tr.iloc[j]
        X, Xt = tr[BASE].values, te[BASE].values
        y, yt = tr['IV'].values, te['IV'].values
        t0 = time.perf_counter()
        m = TabPFNRegressor(device='cpu', random_state=SEED,
                            ignore_pretraining_limits=True).fit(X, y)
        p = m.predict(Xt)
        rows.append(dict(tick=tick, fold=str(qs[i]), model='TabPFN',
                         n_train=len(y), r2=float(r2(yt, p)),
                         secs=time.perf_counter() - t0))
        print(f'{qs[i]}  n={len(y)}  TabPFN {rows[-1]["r2"]:.4f}  '
              f'({rows[-1]["secs"]:.0f}s)', flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)
    print(f'\n{len(rows)} folds at n={CAP} -> {path}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'AAPL')
