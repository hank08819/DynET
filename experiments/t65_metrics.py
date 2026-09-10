"""t65_metrics.py -- 由逐行预测算出各种误差口径, 并在两翼上逐一做配对检验.

t63 现在把每一折每一个模型的逐行预测存进 results/preds/*.npz, 连同实际的隐含
波动率、由该行自己的 S, K, tau 重算的 Vega, 以及区间标签. 这个脚本只读那些文件,
不再拟合任何模型, 因此换一个误差口径不需要重跑实验.

给出的口径:
  R2        区间决定系数, 基准是该区间自己的均值
  MSE, RMSE, MAE, MedAE   隐含波动率的误差
  MSE_px, MAE_px          Vega 加权, 即同样的误差折成价格之后的大小
  MAPE      隐含波动率的相对误差

    python t65_metrics.py                 # 写出 results/metrics_all.csv 并打印两翼
    python t65_metrics.py mae             # 只打印某一个口径
"""
import sys, glob, os
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parents[1]
PRED = HERE / 'results' / 'preds'
REGIONS = ['deep_OTM', 'OTM', 'ATM', 'ITM', 'deep_ITM']
WINGS = ['deep_OTM', 'deep_ITM']
BASE = 'Dynamic ET'
# 名字 -> (函数, 是否越小越好)
METRICS = {
    'r2':     (lambda y, p, v: 1 - ((p - y) ** 2).mean() / y.var(), False),
    'mse':    (lambda y, p, v: ((p - y) ** 2).mean(), True),
    'rmse':   (lambda y, p, v: float(np.sqrt(((p - y) ** 2).mean())), True),
    'mae':    (lambda y, p, v: np.abs(p - y).mean(), True),
    'medae':  (lambda y, p, v: float(np.median(np.abs(p - y))), True),
    'mape':   (lambda y, p, v: (np.abs(p - y) / np.maximum(y, 1e-6)).mean(), True),
    'mse_px': (lambda y, p, v: ((v * (p - y)) ** 2).mean(), True),
    'mae_px': (lambda y, p, v: np.abs(v * (p - y)).mean(), True),
}


def build():
    rows = []
    for f in sorted(glob.glob(str(PRED / '*.npz'))):
        tick, fold, group = Path(f).stem.split('_')
        z = np.load(f, allow_pickle=False)
        y, vega, reg = z['y'].astype(float), z['vega'].astype(float), z['region']
        for name in z.files:
            if name in ('y', 'vega', 'region'): continue
            p = z[name].astype(float)
            for r in REGIONS:
                k = reg == r
                if int(k.sum()) < 30 or y[k].var() < 1e-12: continue
                rec = dict(tick=tick, fold=fold, group=group, model=name, region=r,
                           n=int(k.sum()), var_y=float(y[k].var()))
                for mname, (fn, _) in METRICS.items():
                    rec[mname] = float(fn(y[k], p[k], vega[k]))
                rows.append(rec)
    t = pd.DataFrame(rows)
    t['unit'] = t.tick + '-' + t.fold
    t.to_csv(HERE / 'results' / 'metrics_all.csv', index=False)
    return t


def wing_table(t, region, metric):
    fn, lower_better = METRICS[metric][0], METRICS[metric][1]
    s = t[t.region == region].pivot_table(index='unit', columns='model', values=metric)
    rows = []
    for m in s.columns:
        if m == BASE: continue
        d = (s[m] - s[BASE]) if lower_better else (s[BASE] - s[m])
        d = d.dropna()
        if len(d) < 40: continue
        rows.append(dict(model=m, mean=d.mean(), median=d.median(),
                         ratio=float((s[m] / s[BASE]).dropna().median()),
                         win=int((d > 0).sum()), n=len(d),
                         p=wilcoxon(d, alternative='greater').pvalue))
    o = pd.DataFrame(rows)
    o['holm'] = multipletests(o.p, method='holm')[1]
    return o.sort_values('median', ascending=False)


def main():
    t = build()
    print(f'{len(t)} 行, {t.unit.nunique()} 折, {t.model.nunique()} 个模型, '
          f'{len(METRICS)} 个口径 -> results/metrics_all.csv\n')
    want = [a for a in sys.argv[1:] if a in METRICS] or list(METRICS)
    for metric in want:
        for r in WINGS:
            o = wing_table(t, r, metric)
            if len(o) == 0: continue
            arrow = '越小越好' if METRICS[metric][1] else '越大越好'
            print(f'===== {r} / {metric} ({arrow}); 差值为对手减我们, 比值为对手比我们 =====')
            print(o.round(6).to_string(index=False), '\n')


if __name__ == '__main__':
    main()
