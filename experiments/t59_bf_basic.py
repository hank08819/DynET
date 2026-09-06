"""t59_bf_basic.py -- 正文那一节的正确口径: 我们两步都用, Block Forests 用数据本身的七列.

第一步的 feature enrichment 是本方法的组成部分, 不是可以借给对照的公共输入.
先前的 t28 让 Block Forests 也吃增广后的十五列, 那等于把本方法的第一步送给对手,
比较从设计上就不成立. 这里改正:

  Dynamic ET   basic feature enrichment 的十五列, 按块定预算 max_features = M/d
  ET-enr       同样十五列, 每个节点抽全部候选 (只隔离第二步)
  Block Forests 七个合约与报价列, 语义分组为合约几何六列与流动性一列

三条 Block Forests 变体与包发布的一致: BF-varsel 按语义块袋外调每块候选数,
BF-default 每块以二分之一概率保留后再做 BlockVarSel, BF-corr 把同样的抽样
用在相关块上. 全部 1000 棵树, 30000 训练行, 15000 测试行, 十六个 AAPL 季度.
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
from dynET import augment_basic, block_labels
from blocked_et import BlockedExtraTrees

MAX_TR, MAX_TE, SEED, NT = 30_000, 15_000, 42, 1000
SEM7 = np.array([0, 0, 0, 0, 0, 0, 1])          # 合约几何 6 列 | 流动性 1 列
HERE = Path(__file__).resolve().parents[1]


def main():
    d = load(); d['q'] = d['date'].dt.to_period('Q')
    qs = sorted(d['q'].unique()); rows = []
    path = HERE / 'results' / 'bf_basic7.csv'
    sem = np.unique(SEM7, return_inverse=True)[1]
    mv_sem = np.array([max(1, int(np.sqrt((sem == b).sum())))
                       for b in range(sem.max() + 1)])

    for i in range(4, len(qs)):
        tr = d[d['q'].isin(qs[i-4:i])]; te = d[d['q'] == qs[i]]
        if len(tr) < 500 or len(te) < 50: continue
        rng = np.random.default_rng(SEED)
        if len(tr) > MAX_TR: tr = tr.iloc[rng.choice(len(tr), MAX_TR, replace=False)]
        if len(te) > MAX_TE: te = te.iloc[rng.choice(len(te), MAX_TE, replace=False)]
        B, Bte = tr[BASE].values, te[BASE].values
        X, Xte = augment_basic(B), augment_basic(Bte)
        ytr, yte = tr['IV'].values, te['IV'].values
        dd = X.shape[1]; cb = block_labels(X); M = int(cb.max()) + 1
        cb7 = block_labels(B)
        mv_cor7 = np.array([max(1, int(np.sqrt((cb7 == b).sum())))
                            for b in range(cb7.max() + 1)])
        fold = str(qs[i]); t0 = time.perf_counter()

        arms = {}
        arms['Dynamic ET'] = (ExtraTreesRegressor(
            n_estimators=NT, max_depth=None, min_samples_leaf=1,
            max_features=M / dd, bootstrap=False, n_jobs=-1,
            random_state=SEED).fit(X, ytr).predict(Xte), 15)
        arms['ET-enr'] = (ExtraTreesRegressor(
            n_estimators=NT, max_depth=None, min_samples_leaf=1,
            max_features=1.0, bootstrap=False, n_jobs=-1,
            random_state=SEED).fit(X, ytr).predict(Xte), 15)
        arms['BF-varsel'] = (BlockedExtraTrees(
            NT, mode=2, mvec=mv_sem, blocks=sem).fit(B, ytr).predict(Bte), 7)
        arms['BF-default'] = (BlockedExtraTrees(
            NT, mode=4, mvec=mv_sem, blocks=sem).fit(B, ytr).predict(Bte), 7)
        arms['BF-corr'] = (BlockedExtraTrees(
            NT, mode=4, mvec=mv_cor7, blocks=cb7).fit(B, ytr).predict(Bte), 7)

        line = []
        for name, (p, dcol) in arms.items():
            p = np.asarray(p, dtype=float)
            rows.append(dict(fold=fold, model=name, d=dcol, M_blocks=M,
                             r2=r2_score(yte, p), mae=mean_absolute_error(yte, p)))
            line.append(f'{name} {rows[-1]["r2"]:.4f}')
        print(f'{fold}  M={M}/{dd}  ' + '  '.join(line) +
              f'  ({time.perf_counter()-t0:.0f}s)', flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)

    t = pd.DataFrame(rows)
    print(f'\n{len(t)} fits over {t.fold.nunique()} folds -> {path}\n')
    print(t.groupby('model')[['r2', 'mae']].mean()
           .sort_values('r2', ascending=False)
           .to_string(float_format=lambda x: f'{x:.4f}'))
    piv = t.pivot(index='fold', columns='model', values='r2')
    print()
    for m in piv.columns:
        if m == 'Dynamic ET': continue
        e = (piv['Dynamic ET'] - piv[m]).dropna()
        print(f'  vs {m:<12} dR2={e.mean():+.4f}  {int((e>0).sum())}/{len(e)}'
              f'  p={wilcoxon(e, alternative="greater").pvalue:.4g}')


if __name__ == '__main__':
    main()
