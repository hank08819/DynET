"""t73_block_order.py -- 块按分裂价值的实际次序, 以及由它算出的碰撞概率.

引理 4 的 pi_j 依赖块按"分裂价值"排序后的累积大小, 不是按块大小排序. 把块大小
直接代进去, 等于假定最大的块就是最有价值的块. 这一支不假定, 而是测量:

对每一折的根节点, 按 Extra Trees 的分裂准则给每一列打分(单次随机阈值下的方差
减少量的期望, 用若干次随机阈值取平均), 块的价值取该块内最高的列分, 再按块价值
排序, 用真实次序算 C(M) 与最优块遗漏概率.

同时给出把块大小降序当作价值次序时的数值, 两者并列, 差距有多大一目了然.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from math import comb
from t4_clean_sweep import augment, blocks, BASE
from t24_underlyings import load, PANELS, DATA
from dynET import block_labels

MAX_TR, MAX_TE, SEED, NDRAW = 30_000, 15_000, 42, 200
HERE = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(SEED)


def col_scores(X, y, ndraw=NDRAW):
    """expected variance reduction of one uniform random threshold, per column."""
    n = len(y); tot = y.var() * n
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        x = X[:, j]; lo, hi = x.min(), x.max()
        if hi <= lo: continue
        ts = RNG.uniform(lo, hi, ndraw)
        gains = np.empty(ndraw)
        order = np.argsort(x); xs = x[order]; ys = y[order]
        cs = np.concatenate([[0], np.cumsum(ys)]); cs2 = np.concatenate([[0], np.cumsum(ys ** 2)])
        idx = np.searchsorted(xs, ts)
        for k, i in enumerate(idx):
            if i == 0 or i == n: gains[k] = 0.0; continue
            sl, sr = cs[i], cs[-1] - cs[i]
            ql, qr = cs2[i], cs2[-1] - cs2[i]
            vl = ql - sl * sl / i; vr = qr - sr * sr / (n - i)
            gains[k] = tot - (vl + vr)
        out[j] = gains.mean()
    return out


def C_of(sizes, m):
    d = sum(sizes); Q = np.concatenate([[0], np.cumsum(sizes)])
    pi = []
    for j in range(len(sizes)):
        a = comb(d - Q[j], m) if d - Q[j] >= m else 0
        b = comb(d - Q[j + 1], m) if d - Q[j + 1] >= m else 0
        pi.append((a - b) / comb(d, m))
    miss = comb(d - sizes[0], m) / comb(d, m) if d - sizes[0] >= m else 0.0
    return float(sum(p * p for p in pi)), float(miss)


def main(ticks):
    rows = []
    for tick in ticks:
        d = load(DATA / PANELS[tick]); d['q'] = d['date'].dt.to_period('Q')
        qs = sorted(d['q'].unique())
        for i in range(4, len(qs)):
            tr = d[d['q'].isin(qs[i-4:i])]
            if len(tr) < 500: continue
            rng = np.random.default_rng(SEED)
            if len(tr) > MAX_TR: tr = tr.iloc[rng.choice(len(tr), MAX_TR, replace=False)]
            A = augment(tr[BASE].values); y = tr['IV'].values
            lab = block_labels(A); M = int(lab.max()) + 1; dd = A.shape[1]
            sc = col_scores(A, y)
            val = np.array([sc[lab == b].max() for b in range(M)])
            size = np.array([(lab == b).sum() for b in range(M)])
            by_value = size[np.argsort(-val)]
            by_size = np.sort(size)[::-1]
            m = M
            Cv, missv = C_of(list(by_value), m)
            Cs, misss = C_of(list(by_size), m)
            curve = {f'C_m{m_}': C_of(list(by_value), m_)[0] for m_ in range(1, dd + 1)}
            rows.append(dict(tick=tick, fold=str(qs[i]), M=M, d=dd,
                             order=' '.join(map(str, by_value)), **curve,
                             top_size_by_value=int(by_value[0]),
                             top_size_by_size=int(by_size[0]),
                             C_by_value=Cv, miss_by_value=missv,
                             C_by_size=Cs, miss_by_size=misss,
                             best_is_largest=bool(by_value[0] == by_size[0])))
            print(f"{tick} {qs[i]}  M={M}  价值序 {tuple(by_value)}  C={Cv:.4f} "
                  f"| 大小序 {tuple(by_size)}  C={Cs:.4f}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(HERE / 'results' / 'block_order.csv', index=False)
    print(f"\n{len(t)} folds -> results/block_order.csv")
    print(t[['C_by_value', 'C_by_size', 'miss_by_value', 'miss_by_size']].mean().round(4).to_string())
    print('最有价值的块同时也是最大的块:', int(t.best_is_largest.sum()), '/', len(t))


if __name__ == '__main__':
    main(sys.argv[1:] or ['AAPL', 'TSLA', 'NVDA', 'SPY', 'QQQ'])
