"""t81_export_wide.py -- 把 48 折按十五列与十八列导出, 附相关块标签, 供 CRAN 包读取.

blockForest 需要一个块划分作为输入. 这里给它的是本方法第二步推出的相关块, 也就是
说前两步都交给它, 只留第三步(怎么花预算)不同. 每折另存一个块向量文件.
"""
import os, sys, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
for _m in ('bottleneck', 'numexpr'):
    sys.modules.setdefault(_m, None)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd

MAX_TR, MAX_TE, SEED = 30_000, 15_000, 42
BASE7 = ['m', 'logm', 'T', 'K', 'S', 'is_call', 'spr']
DATA = Path(os.environ.get('OPTDATA',
            '/Users/henry_han/LHP/JMP1+MS2027/data/OptionData2026'))
PANELS = {'AAPL': 'aapl_2016_2020', 'NVDA': 'nvda_2020_2022',
          'QQQ': 'qqq_2020_2022', 'SPY': 'spy_2020_2022', 'TSLA': 'tsla_2019_2022'}
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else 'folds_wide')
sys.path.append(str(Path(__file__).resolve().parents[1] / 'dynet-release' / 'experiments'))
exec(open(Path(__file__).resolve().parents[1] / 'code' / 't80_bf_widths.py').read()
     .split('def main(')[0].split("from blocked_et import")[1].split('\n', 1)[1])

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for tick, stem in PANELS.items():
        d = pd.read_parquet(DATA / (stem + '.parquet'))
        d['q'] = d['date'].dt.to_period('Q')
        qs = sorted(d['q'].unique())
        for i in range(4, len(qs)):
            tr = d[d['q'].isin(qs[i-4:i])]; te = d[d['q'] == qs[i]]
            if len(tr) < 500 or len(te) < 50: continue
            rng = np.random.default_rng(SEED)
            if len(tr) > MAX_TR: tr = tr.iloc[rng.choice(len(tr), MAX_TR, replace=False)]
            if len(te) > MAX_TE: te = te.iloc[rng.choice(len(te), MAX_TE, replace=False)]
            B, Bte = tr[BASE7].values, te[BASE7].values
            for w, aug in ((15, augment_basic), (18, augment_shape18)):
                X, Xt = aug(B), aug(Bte)
                cb = block_labels(X)
                cols = [f'x{j}' for j in range(X.shape[1])]
                stemf = f'{tick}_{qs[i]}_w{w}'
                pd.DataFrame(X, columns=cols).assign(IV=tr['IV'].values)\
                    .to_csv(OUT / f'{stemf}_train.csv', index=False)
                pd.DataFrame(Xt, columns=cols).assign(IV=te['IV'].values)\
                    .to_csv(OUT / f'{stemf}_test.csv', index=False)
                np.savetxt(OUT / f'{stemf}_blocks.txt', cb + 1, fmt='%d')
            n += 1
            print(f'{tick} {qs[i]}  导出 15/18 列', flush=True)
    print(f'\n{n} 折 -> {OUT}')

if __name__ == '__main__':
    main()
