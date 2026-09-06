"""t55_greek_isolation.py -- Greeks 的识别、隔离与恢复, 在完整面板上.

三件事, 都在 AAPL 2016-2020 的全部合约边上做, 不再用文件前缀:

  1. 重算. 用面板自己的 S, K, tau 和供应商的隐含波动率, 按 Black-Scholes 重算
     delta, gamma, vega, 与供应商发布的列比较.
  2. 准入. 供应商列里不可能的取值(负 vega, 负 gamma)先计数再剔除, 两种口径的
     相关度分别报出, 秩相关对这些取值不敏感, 线性相关敏感.
  3. 反演. 全局恒等式 sigma = vega / (S^2 tau gamma) 在统一单位下成立;
     这里报出它在供应商数据上的实际恢复精度.

写出 results/greek_isolation.csv, 图与表由它生成.
"""
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import os
DATA_DIR = os.environ.get('DYNET_DATA', os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), 'data'))
import numpy as np, pandas as pd
from scipy.stats import norm

PANEL = Path(DATA_DIR) / 'aapl_2016_2020.csv'
HERE = Path(__file__).resolve().parents[1]
VEGA_PER_POINT = 100.0        # 供应商的 vega 是每一个波动率百分点


def load():
    raw = pd.read_csv(PANEL, low_memory=False, on_bad_lines='skip')
    raw.columns = [c.strip().strip('[]') for c in raw.columns]
    n = lambda c: pd.to_numeric(raw[c], errors='coerce')
    fr = []
    for t, call in (('C', True), ('P', False)):
        fr.append(pd.DataFrame({
            'S': n('UNDERLYING_LAST'), 'K': n('STRIKE'), 'T': n('DTE') / 365.,
            'iv': n(f'{t}_IV'), 'delta': n(f'{t}_DELTA'), 'gamma': n(f'{t}_GAMMA'),
            'vega': n(f'{t}_VEGA'), 'call': call}))
    d = pd.concat(fr, ignore_index=True).dropna()
    d = d[(d.iv > .01) & (d.iv < 4) & (d['T'] > 1 / 365) & (d.K > 0) & (d.S > 0)]
    return d[d.vega.abs() > 1e-8].reset_index(drop=True)


def main():
    d = load()
    S, K, T, s = d.S.values, d.K.values, d['T'].values, d.iv.values
    d1 = (np.log(S / K) + (s ** 2 / 2) * T) / (s * np.sqrt(T))
    rec = {'delta': np.where(d.call.values, norm.cdf(d1), norm.cdf(d1) - 1),
           'gamma': norm.pdf(d1) / (S * s * np.sqrt(T)),
           'vega': S * norm.pdf(d1) * np.sqrt(T) / VEGA_PER_POINT}

    ok = (d.vega > 0) & (d.gamma > 0) & (d.delta.abs() <= 1)
    rows = []
    for k in ('delta', 'gamma', 'vega'):
        a, b = rec[k], d[k].values
        rows.append(dict(
            column=k, rows=len(d), rows_admissible=int(ok.sum()),
            pearson_all=np.corrcoef(a, b)[0, 1],
            spearman_all=pd.Series(a).corr(pd.Series(b), method='spearman'),
            pearson_admissible=np.corrcoef(a[ok.values], b[ok.values])[0, 1]))
    t = pd.DataFrame(rows)

    # 反演: sigma = vega / (S^2 tau gamma), vega 换算到每一个单位波动率
    inv = (d.vega.values * VEGA_PER_POINT) / (S ** 2 * T * d.gamma.values)
    m = ok.values & np.isfinite(inv)
    ratio = inv[m] / s[m]
    q = np.percentile(ratio, [1, 25, 50, 75, 99])
    t.attrs = {}
    extra = dict(recovery_median=float(np.median(ratio)),
                 recovery_p1=float(q[0]), recovery_p25=float(q[1]),
                 recovery_p75=float(q[3]), recovery_p99=float(q[4]),
                 recovery_within_1pct=float(np.mean(np.abs(ratio - 1) < .01)),
                 recovery_within_10pct=float(np.mean(np.abs(ratio - 1) < .10)),
                 recovery_spearman=float(pd.Series(inv[m]).corr(pd.Series(s[m]),
                                                                method='spearman')),
                 gamma_decimals=5, gamma_median=float(np.median(d.gamma.values[m])),
                 neg_vega=int((d.vega < 0).sum()), neg_gamma=int((d.gamma < 0).sum()),
                 min_vega=float(d.vega.min()))
    t.to_csv(HERE / 'results' / 'greek_isolation.csv', index=False)
    pd.Series(extra).to_csv(HERE / 'results' / 'greek_recovery.csv', header=False)

    print(t.round(4).to_string(index=False))
    print()
    for k, v in extra.items():
        print(f'  {k:22s} {v}')


if __name__ == '__main__':
    main()
