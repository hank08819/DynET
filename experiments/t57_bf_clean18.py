"""t57_bf_clean18.py -- 无泄漏的十八列上, Dynamic ET 对全部三个 Block Forests 变体.

t28 在十五列上做过同一件事. 这里换成第 III 节给出的无泄漏十二列基础加六项增广,
共十八列: 七个合约与报价列, 固定波动率下的四个敏感度形状, 相对价差, 再加
m*T, logm^2, m*spr, spr/T, m^2, T^2.

Block Forests 需要一个先验的语义分组, 十八列的分组按同样的原则给出:

  0 合约几何      m, logm, T, K, S, is_call
  1 流动性        spr, spr/S
  2 敏感度形状    delta0, gamma0, vega0, theta0
  3 交互项        六项

三个变体与 t28 一致: BF-varsel 按语义块袋外调每块候选数, BF-default 是包发布的
默认(每块以二分之一概率保留后再做 BlockVarSel), BF-corr 把同样的抽样用在相关块上.
协议与正文相同, 只有一处不同: 这一节的五条臂一律用 500 棵树而不是 1000.
Block Forests 的这份实现为了能表达逐块的候选向量而写在 numba 上, 每个节点的
候选构造在 Python 层, 比 scikit-learn 的 Cython 分裂器慢一个量级; 五条臂同时
减半保持了比较本身的公平, 而每折的耗时减半. 精度上 Extra Trees 从 500 棵到
1000 棵的差别在第四位小数.
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
from dynET import base12_leakfree, augment_clean18, block_labels
from blocked_et import BlockedExtraTrees

MAX_TR, MAX_TE, SEED, NT = 30_000, 15_000, 42, 500
#           m logm T  K  S call spr d0 g0 v0 t0 spr/S | 六个交互项
SEM = np.array([0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 1, 3, 3, 3, 3, 3, 3])
HERE = Path(__file__).resolve().parents[1]


def main():
    d = load(); d['q'] = d['date'].dt.to_period('Q')
    qs = sorted(d['q'].unique()); rows = []
    path = HERE / 'results' / 'blockforest18.csv'
    sem = np.unique(SEM, return_inverse=True)[1]
    mv_sem = np.array([max(1, int(np.sqrt((sem == b).sum())))
                       for b in range(sem.max() + 1)])

    for i in range(4, len(qs)):
        tr = d[d['q'].isin(qs[i-4:i])]; te = d[d['q'] == qs[i]]
        if len(tr) < 500 or len(te) < 50: continue
        rng = np.random.default_rng(SEED)
        if len(tr) > MAX_TR: tr = tr.iloc[rng.choice(len(tr), MAX_TR, replace=False)]
        if len(te) > MAX_TE: te = te.iloc[rng.choice(len(te), MAX_TE, replace=False)]
        Xtr = augment_clean18(base12_leakfree(tr[BASE].values))
        Xte = augment_clean18(base12_leakfree(te[BASE].values))
        ytr, yte = tr['IV'].values, te['IV'].values
        dd = Xtr.shape[1]
        cb = block_labels(Xtr); M = int(cb.max()) + 1
        mv_cor = np.array([max(1, int(np.sqrt((cb == b).sum())))
                           for b in range(cb.max() + 1)])
        fold = str(qs[i]); t0 = time.perf_counter()

        arms = {}
        arms['Dynamic ET'] = ExtraTreesRegressor(
            n_estimators=NT, max_depth=None, min_samples_leaf=1,
            max_features=M / dd, bootstrap=False, n_jobs=-1,
            random_state=SEED).fit(Xtr, ytr).predict(Xte)
        arms['ET-enr'] = ExtraTreesRegressor(
            n_estimators=NT, max_depth=None, min_samples_leaf=1,
            max_features=1.0, bootstrap=False, n_jobs=-1,
            random_state=SEED).fit(Xtr, ytr).predict(Xte)
        arms['BF-varsel'] = BlockedExtraTrees(
            NT, mode=2, mvec=mv_sem, blocks=sem).fit(Xtr, ytr).predict(Xte)
        arms['BF-default'] = BlockedExtraTrees(
            NT, mode=4, mvec=mv_sem, blocks=sem).fit(Xtr, ytr).predict(Xte)
        arms['BF-corr'] = BlockedExtraTrees(
            NT, mode=4, mvec=mv_cor, blocks=cb).fit(Xtr, ytr).predict(Xte)

        line = []
        for name, p in arms.items():
            p = np.asarray(p, dtype=float)
            rows.append(dict(fold=fold, model=name, M_blocks=M, d=dd,
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
