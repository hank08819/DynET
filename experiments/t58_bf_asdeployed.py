"""t58_bf_asdeployed.py -- 各自按自己的部署方式: Dynamic ET 十八列, Block Forests 十二列.

t57 让五条臂看同一个十八列矩阵, 回答的是"同等信息下预算规则谁强". 这一支回答另一个
问题: 两种方法各自按它实际会被使用的样子摆出来, 结果如何.

  Dynamic ET    Greek leak-free enrichment 的十八列, 1000 棵树
  Block Forests 七个合约与报价列, 语义分组为合约几何六列与流动性一列, 500 棵树

七列是这份数据里不含 Greeks 又不泄漏价格水平的全部原始信息: 中间价与最后成交价
是隐含波动率的反解来源, 因此只有价差可用. 增广是本方法的一部分, 所以 Block
Forests 拿到的是未增广的原始列; 树数按各自的宽度取值, 两个数都在表里写明.
这一支不是同等信息的比较, 与 t57 并列给出.
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

MAX_TR, MAX_TE, SEED = 30_000, 15_000, 42
NT_OURS, NT_BF = 1000, 500
SEM7 = np.array([0, 0, 0, 0, 0, 0, 1])       # 合约几何 6 列 | 流动性 1 列
HERE = Path(__file__).resolve().parents[1]


def main():
    d = load(); d['q'] = d['date'].dt.to_period('Q')
    qs = sorted(d['q'].unique()); rows = []
    path = HERE / 'results' / 'bf_asdeployed.csv'
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
        X = augment_clean18(base12_leakfree(B))
        Xte = augment_clean18(base12_leakfree(Bte))
        ytr, yte = tr['IV'].values, te['IV'].values
        dd = X.shape[1]; cb = block_labels(X); M = int(cb.max()) + 1
        cb7 = block_labels(B); mv_cor7 = np.array(
            [max(1, int(np.sqrt((cb7 == b).sum()))) for b in range(cb7.max() + 1)])
        fold = str(qs[i]); t0 = time.perf_counter()

        arms = {}
        arms['Dynamic ET, 18 col'] = (ExtraTreesRegressor(
            n_estimators=NT_OURS, max_depth=None, min_samples_leaf=1,
            max_features=M / dd, bootstrap=False, n_jobs=-1,
            random_state=SEED).fit(X, ytr).predict(Xte), NT_OURS, dd)
        arms['BF-varsel, 7 col'] = (BlockedExtraTrees(
            NT_BF, mode=2, mvec=mv_sem, blocks=sem).fit(B, ytr).predict(Bte), NT_BF, 7)
        arms['BF-default, 7 col'] = (BlockedExtraTrees(
            NT_BF, mode=4, mvec=mv_sem, blocks=sem).fit(B, ytr).predict(Bte), NT_BF, 7)
        arms['BF-corr, 7 col'] = (BlockedExtraTrees(
            NT_BF, mode=4, mvec=mv_cor7, blocks=cb7).fit(B, ytr).predict(Bte), NT_BF, 7)

        line = []
        for name, (p, nt, dcol) in arms.items():
            p = np.asarray(p, dtype=float)
            rows.append(dict(fold=fold, model=name, trees=nt, d=dcol, M_blocks=M,
                             r2=r2_score(yte, p), mae=mean_absolute_error(yte, p)))
            line.append(f'{name} {rows[-1]["r2"]:.4f}')
        print(f'{fold}  ' + '  '.join(line) + f'  ({time.perf_counter()-t0:.0f}s)',
              flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)

    t = pd.DataFrame(rows)
    print(f'\n{len(t)} fits over {t.fold.nunique()} folds -> {path}\n')
    print(t.groupby('model')[['r2', 'mae']].mean()
           .sort_values('r2', ascending=False)
           .to_string(float_format=lambda x: f'{x:.4f}'))
    piv = t.pivot(index='fold', columns='model', values='r2')
    base = 'Dynamic ET, 18 col'
    print()
    for m in piv.columns:
        if m == base: continue
        e = (piv[base] - piv[m]).dropna()
        print(f'  vs {m:<20} dR2={e.mean():+.4f}  {int((e>0).sum())}/{len(e)}'
              f'  p={wilcoxon(e, alternative="greater").pvalue:.4g}')


if __name__ == '__main__':
    main()
