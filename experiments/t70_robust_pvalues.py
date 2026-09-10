"""t70_robust_pvalues.py -- 折与折不独立时的稳健显著性.

四季度滚动窗使相邻折的训练集重叠四分之三, 五个标的又共享全市场冲击, 所以 48 折
不是 48 个独立观测, 独立同分布假定下的 Wilcoxon 会低估 p 值. 这一支给出三个不
依赖该假定的口径:

  1. 移动块自助: 在每个标的内部按长度 L 的连续折块重采样, 重建 48 折的均值差,
     块长取 4, 与训练窗的重叠长度一致;
  2. 标的层的符号翻转置换: 把一个标的的全部折一起翻转符号, 五个标的共 32 种,
     这是把"整张面板都是巧合"当作原假设的最保守读法;
  3. 以标的为簇的自助: 有放回地抽标的, 给出簇稳健的区间.

三个口径都比独立 Wilcoxon 保守, 报告时以最保守的一个为准.
"""
import sys, glob, os
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from itertools import product

HERE = Path(__file__).resolve().parents[1]
RES = HERE / 'results'
BASE = 'Dynamic ET'
B = 20000
BLOCK = 4
RNG = np.random.default_rng(42)


def load():
    """overall fold-level R^2 per model, protocol seed.

    Each comparator is paired with the Dynamic ET fitted in the same run, so the
    difference never mixes two runs; the two runs differ only in the block
    routine used and agree to $0.001$ on the shared method.
    """
    fr = []
    a = pd.read_csv(RES / 'seeds7.csv').dropna(subset=['r2'])
    more = RES / 'seeds_more.csv'
    if more.exists():
        a = pd.concat([a, pd.read_csv(more).dropna(subset=['r2'])], ignore_index=True)
    a = a[a.seed == 42][['fold', 'model', 'r2']].assign(tick='AAPL')
    a['model'] = a.model.replace({'ET-aug': 'ET-enr'})
    fr.append(a)
    for f in sorted(glob.glob(str(RES / 'seeds_*_*.csv'))):
        if os.path.basename(f).startswith('seeds7'): continue
        t = pd.read_csv(f).dropna(subset=['r2'])
        if 'tick' in t and 'seed' in t:
            fr.append(t[t.seed == 42][['tick', 'fold', 'model', 'r2']])
    for f in sorted(glob.glob(str(RES / 'bf_basic7_*.csv'))):
        t = pd.read_csv(f)
        if 'tick' not in t: t['tick'] = Path(f).stem.split('_')[-1]
        t = t[['tick', 'fold', 'model', 'r2']].copy()
        t['model'] = t.model.where(t.model == BASE, t.model + ' [bf run]')
        t['model'] = t.model.where(t.model != BASE, BASE + ' [bf run]')
        fr.append(t)
    return pd.concat(fr, ignore_index=True)


def moving_block(d_by_tick):
    """one bootstrap replicate: within each ticker, resample contiguous blocks."""
    out = []
    for v in d_by_tick:
        n = len(v)
        if n <= BLOCK:
            out.append(v[RNG.integers(0, n, n)])
            continue
        k = int(np.ceil(n / BLOCK))
        starts = RNG.integers(0, n - BLOCK + 1, k)
        out.append(np.concatenate([v[s:s + BLOCK] for s in starts])[:n])
    return np.concatenate(out)


def signflip_p(d_by_tick):
    """flip the sign of a whole ticker at a time; 2^5 = 32 assignments."""
    obs = np.mean(np.concatenate(d_by_tick))
    n = [len(v) for v in d_by_tick]; tot = sum(n)
    means = []
    for signs in product([1, -1], repeat=len(d_by_tick)):
        means.append(sum(s * v.sum() for s, v in zip(signs, d_by_tick)) / tot)
    means = np.array(means)
    return float((means >= obs - 1e-12).mean()), obs


def main():
    t = load()
    piv = t.pivot_table(index=['tick', 'fold'], columns='model', values='r2')
    comps = [m for m in piv.columns if not m.startswith(BASE)]
    rows = []
    for m in comps:
        ref = BASE + ' [bf run]' if m.endswith('[bf run]') else BASE
        d = (piv[ref] - piv[m]).dropna()
        if len(d) < 20: continue
        by = [d.xs(k, level=0).sort_index().values for k in sorted(set(d.index.get_level_values(0)))]
        boot = np.array([moving_block(by).mean() for _ in range(B)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p_boot = float((boot <= 0).mean())
        p_flip, obs = signflip_p(by)
        cl = np.array([v.mean() for v in by])
        cboot = np.array([RNG.choice(cl, len(cl), replace=True).mean() for _ in range(B)])
        rows.append(dict(model=m.replace(' [bf run]',''), mean=obs, folds=len(d),
                         wilcoxon=wilcoxon(d, alternative='greater').pvalue,
                         block_lo=lo, block_hi=hi, p_block=p_boot,
                         p_signflip=p_flip,
                         cluster_lo=np.percentile(cboot, 2.5),
                         cluster_hi=np.percentile(cboot, 97.5)))
    o = pd.DataFrame(rows).sort_values('mean')
    o.to_csv(RES / 'robust_pvalues.csv', index=False)
    pd.set_option('display.width', 200)
    print(o.round(5).to_string(index=False))
    print('\n符号翻转的最小可能 p 值 = 1/32 = 0.031 (五个标的)')


if __name__ == '__main__':
    main()
