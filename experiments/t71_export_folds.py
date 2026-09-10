"""t71_export_folds.py -- 把每一折的训练与测试行导出成 CSV, 供 R 版 blockForest 使用.

审稿意见要求用发表的 R 包原样跑一次 Block Forests, 而不是我们的重实现. 为了让两边
读到的是同一批行, 折的切分与抽样在 Python 这一侧完成一次, 落盘之后 R 只负责拟合。

导出的是数据自带的七列与目标, 不含任何增广列: Block Forests 读七列, 这是正文规定的
口径, 增广是本方法自己的第一步.
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t4_clean_sweep import BASE
from t24_underlyings import load, PANELS, DATA

MAX_TR, MAX_TE, SEED = 30_000, 15_000, 42
HERE = Path(__file__).resolve().parents[1]
OUT = HERE / 'folds'


def main(ticks):
    OUT.mkdir(exist_ok=True)
    n = 0
    for tick in ticks:
        d = load(DATA / PANELS[tick]); d['q'] = d['date'].dt.to_period('Q')
        qs = sorted(d['q'].unique())
        for i in range(4, len(qs)):
            tr = d[d['q'].isin(qs[i-4:i])]; te = d[d['q'] == qs[i]]
            if len(tr) < 500 or len(te) < 50: continue
            rng = np.random.default_rng(SEED)
            if len(tr) > MAX_TR: tr = tr.iloc[rng.choice(len(tr), MAX_TR, replace=False)]
            if len(te) > MAX_TE: te = te.iloc[rng.choice(len(te), MAX_TE, replace=False)]
            fold = str(qs[i])
            for part, frame in (('train', tr), ('test', te)):
                out = frame[BASE].copy(); out['IV'] = frame['IV'].values
                out.to_csv(OUT / f'{tick}_{fold}_{part}.csv', index=False)
            n += 1
            print(f'{tick} {fold}  train={len(tr)}  test={len(te)}', flush=True)
    print(f'\n{n} folds -> {OUT}')


if __name__ == '__main__':
    main(sys.argv[1:] or ['AAPL', 'TSLA', 'NVDA', 'SPY', 'QQQ'])
