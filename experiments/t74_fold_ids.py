"""t74_fold_ids.py -- 每一折抽到哪些行, 落盘存档并自校验.

审稿意见要求能核对"各方法读到的是不是同一批行". 抽样在协议里是确定的: 给定面板、
折与种子, 行的下标就定了. 这一支把这些下标写出来, 并附一个校验和, 使得任何人都能
在不重跑训练的情况下, 确认两支实验用的是不是同一批行.

同时核对一件事: 训练切片在抽样之前有没有被再排序一次. pandas 的 sort_values 默认
不是稳定排序, 对一个已经按日期排好的切片再排一次, 会打乱同一天之内的次序, 从而
改变随后抽到的行. 本脚本把两种做法的下标都算出来并比较.

    python t74_fold_ids.py            # 五张面板, 写 results/fold_ids/
"""
import sys, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t24_underlyings import load, PANELS, DATA

MAX_TR, MAX_TE, SEED = 30_000, 15_000, 42
HERE = Path(__file__).resolve().parents[1]
OUT = HERE / 'results' / 'fold_ids'


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(np.sort(a)).tobytes()).hexdigest()[:16]


def main(ticks):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for tick in ticks:
        d = load(DATA / PANELS[tick]); d['q'] = d['date'].dt.to_period('Q')
        qs = sorted(d['q'].unique())
        for i in range(4, len(qs)):
            tr = d[d['q'].isin(qs[i-4:i])]; te = d[d['q'] == qs[i]]
            if len(tr) < 500 or len(te) < 50: continue
            fold = str(qs[i])
            rng = np.random.default_rng(SEED)
            itr = tr.iloc[rng.choice(len(tr), MAX_TR, replace=False)].index.values \
                if len(tr) > MAX_TR else tr.index.values
            ite = te.iloc[rng.choice(len(te), MAX_TE, replace=False)].index.values \
                if len(te) > MAX_TE else te.index.values
            # 再排序一次之后会抽到什么, 用来核对两支实验是否一致
            tr2 = tr.sort_values('date')
            rng2 = np.random.default_rng(SEED)
            itr2 = tr2.iloc[rng2.choice(len(tr2), MAX_TR, replace=False)].index.values \
                if len(tr2) > MAX_TR else tr2.index.values
            np.savez_compressed(OUT / f'{tick}_{fold}.npz', train=itr, test=ite)
            rows.append(dict(tick=tick, fold=fold, n_train=len(itr), n_test=len(ite),
                             train_sha=digest(itr), test_sha=digest(ite),
                             train_sha_resorted=digest(itr2),
                             overlap_resorted=len(set(itr) & set(itr2))))
            print(f'{tick} {fold}  train {len(itr)}  test {len(ite)}  '
                  f'sha {rows[-1]["train_sha"]}', flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(HERE / 'results' / 'fold_ids.csv', index=False)
    same = int((t.train_sha == t.train_sha_resorted).sum())
    print(f'\n{len(t)} folds -> {OUT} and results/fold_ids.csv')
    print(f'再排序后抽到同一批行的折: {same}/{len(t)}; '
          f'重叠行数中位 {int(t.overlap_resorted.median())} / {MAX_TR}')


if __name__ == '__main__':
    main(sys.argv[1:] or ['AAPL', 'TSLA', 'NVDA', 'SPY', 'QQQ'])
