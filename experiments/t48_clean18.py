"""t48_clean18.py -- 无泄漏的十八列增广与十五列基本增广的对照.

十二列那套(full20)带 Greek 泄漏, 不能进正文. 这里给出同样宽度、但不读目标的
版本: 七个合约与报价列, 加上固定波动率下的四个敏感度形状, 加上相对价差, 共
十二列基础, 再接六个交互项, 共十八列. 敏感度只以它对合约的依赖进入, 使它成为
答案的那个波动率被固定成常数, 所以命题 1 对它不成立.

两条臂都用 Dynamic ET 的规则, 也各跑一条全候选消融, 折数与协议同正文.
"""
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import wilcoxon
from t4_clean_sweep import load, BASE
from dynET import augment_basic, augment_shape18, block_count

MAX_TR, MAX_TE, SEED, NT = 30_000, 15_000, 42, 1000
HERE = Path(__file__).resolve().parents[1]


def main():
    d = load(); d['q'] = d['date'].dt.to_period('Q')
    qs = sorted(d['q'].unique()); rows = []
    for i in range(4, len(qs)):
        tr = d[d['q'].isin(qs[i-4:i])]; te = d[d['q'] == qs[i]]
        if len(tr) < 500 or len(te) < 50: continue
        rng = np.random.default_rng(SEED)
        if len(tr) > MAX_TR: tr = tr.iloc[rng.choice(len(tr), MAX_TR, replace=False)]
        if len(te) > MAX_TE: te = te.iloc[rng.choice(len(te), MAX_TE, replace=False)]
        B, Bte = tr[BASE].values, te[BASE].values
        ytr, yte = tr['IV'].values, te['IV'].values
        fold = str(qs[i]); line = []
        for name, fn in (('clean15', augment_basic), ('clean18', augment_shape18)):
            A, Ate = fn(B), fn(Bte)
            dd = A.shape[1]; M = block_count(A)
            for arm, mf in (('Dynamic ET', M / dd), ('ET-aug', 1.0)):
                t0 = time.perf_counter()
                p = ExtraTreesRegressor(n_estimators=NT, max_depth=None,
                        min_samples_leaf=1, max_features=mf, bootstrap=False,
                        n_jobs=-1, random_state=SEED).fit(A, ytr).predict(Ate)
                rows.append(dict(fold=fold, regime=name, arm=arm, d=dd, M=M,
                                 r2=r2_score(yte, p), mae=mean_absolute_error(yte, p),
                                 secs=time.perf_counter()-t0))
                line.append(f'{name}/{arm} {rows[-1]["r2"]:.4f}')
        print(f'{fold}  M={rows[-2]["M"]}/{rows[-2]["d"]}  ' + '  '.join(line), flush=True)
        pd.DataFrame(rows).to_csv(HERE / 'results' / 'clean18.csv', index=False)

    t = pd.DataFrame(rows)
    print('\n== 平均 ==')
    print(t.pivot_table(index='arm', columns='regime', values=['r2', 'secs'])
           .round(4).to_string())
    print('\n块数:', t.groupby('regime')[['M', 'd']].mean().round(2).to_dict())
    piv = t.pivot_table(index='fold', columns=['regime', 'arm'], values='r2')
    a = piv[('clean18', 'Dynamic ET')]; b = piv[('clean15', 'Dynamic ET')]
    print(f'\nDynamic ET  18 列 - 15 列: {(a-b).mean():+.4f}  '
          f'{int((a>b).sum())}/{len(a)} 折  p={wilcoxon(a, b)[1]:.4f}')
    for r in ('clean15', 'clean18'):
        x = piv[(r, 'Dynamic ET')]; y = piv[(r, 'ET-aug')]
        print(f'{r}: 规则 - 全候选 {(x-y).mean():+.4f}  {int((x>y).sum())}/{len(x)}  '
              f'p={wilcoxon(x, y)[1]:.4f}')


if __name__ == '__main__':
    main()
