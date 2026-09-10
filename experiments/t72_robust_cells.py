"""t72_robust_cells.py -- 两翼 240 个格子的移动块自助.

t70 只对整体比较做了不依赖独立性的检验; 审稿意见指出正文那句"两翼所有比较在这些
程序下结论相同"没有逐格支撑. 这一支把移动块自助搬到每一个格子上:
十五个对手 x 两个区间 x 八个误差口径 = 240 个比较.

块长取 4, 与四季度训练窗的重叠一致, 在每个标的内部重采样连续折块.
越小越好的口径自动取反, 所以每一格的差值都是"我们更好为正".
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parents[1]
BASE = 'Dynamic ET'
WINGS = ['deep_OTM', 'deep_ITM']
LOWER_BETTER = {'mse', 'rmse', 'mae', 'medae', 'mape', 'mse_px', 'mae_px'}
METRICS = ['r2', 'mse', 'rmse', 'mae', 'medae', 'mape', 'mse_px', 'mae_px']
BLOCK, B = 4, 4000
RNG = np.random.default_rng(42)


def block_boot(by_tick):
    out = []
    for v in by_tick:
        n = len(v)
        if n <= BLOCK:
            out.append(v[RNG.integers(0, n, n)]); continue
        k = int(np.ceil(n / BLOCK))
        st = RNG.integers(0, n - BLOCK + 1, k)
        out.append(np.concatenate([v[s:s + BLOCK] for s in st])[:n])
    return np.concatenate(out).mean()


def main():
    t = pd.read_csv(HERE / 'results' / 'metrics_all.csv')
    t['unit'] = t.tick + '-' + t.fold
    rows = []
    for wing in WINGS:
        s = t[t.region == wing]
        for metric in METRICS:
            piv = s.pivot_table(index='unit', columns='model', values=metric)
            for m in piv.columns:
                if m == BASE: continue
                d = (piv[m] - piv[BASE]) if metric in LOWER_BETTER else (piv[BASE] - piv[m])
                d = d.dropna()
                if len(d) < 40: continue
                by = [d[[u.startswith(k) for u in d.index]].values
                      for k in ('AAPL', 'TSLA', 'NVDA', 'SPY', 'QQQ')]
                by = [v for v in by if len(v)]
                boot = np.array([block_boot(by) for _ in range(B)])
                lo, hi = np.percentile(boot, [2.5, 97.5])
                rows.append(dict(wing=wing, metric=metric, model=m,
                                 mean=float(d.mean()), lo=lo, hi=hi,
                                 above_zero=bool(lo > 0), folds=len(d)))
    o = pd.DataFrame(rows)
    o.to_csv(HERE / 'results' / 'robust_cells.csv', index=False)
    n = len(o); k = int(o.above_zero.sum())
    print(f'{n} 个格子, 区间下界大于零的 {k} 个 ({100*k/n:.1f}%)')
    print('\n未过的格子:')
    print(o[~o.above_zero][['wing', 'metric', 'model', 'mean', 'lo', 'hi']]
          .round(5).to_string(index=False))
    print('\n按口径:')
    print(o.groupby('metric').above_zero.agg(['sum', 'size']).to_string())


if __name__ == '__main__':
    main()
