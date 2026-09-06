"""t60_tuned18.py -- Greek leak-free enrichment 上的搜索版 Dynamic ET.

与 t18 对十五列做的事完全一样, 只是特征换成第 III 节的十八列: 四十组配置从
tau, 叶大小, 深度上限, 分裂下限的网格里随机抽取, 在训练窗的时间内部划分上打分
(前五分之四拟合, 后五分之一评分, 测试季度从不读取), 胜出者在整个窗口上以全尺寸
重新拟合. 同时记录未搜索的规则, 两行的差就是搜索值多少.
"""
import sys, time, itertools, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from t4_clean_sweep import load, BASE
from dynET import base12_leakfree, augment_clean18, block_labels

MAX_TR, MAX_TE, SEED = 30_000, 15_000, 42
N_IN, N_FULL, BUDGET = 300, 1000, 40
GRID = [dict(tau=t, leaf=l, depth=dp, split=sp)
        for t, l, dp, sp in itertools.product(
            [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
            [1, 2, 3, 5], [None, 24], [2, 5])]
HERE = Path(__file__).resolve().parents[1]


def nblocks(X, tau):
    return int(block_labels(X, tau).max()) + 1


def main():
    d = load(); d['q'] = d['date'].dt.to_period('Q')
    qs = sorted(d['q'].unique()); rows = []
    path = HERE / 'results' / 'tuned18.csv'
    for i in range(4, len(qs)):
        tr = d[d['q'].isin(qs[i-4:i])].sort_values('date')
        te = d[d['q'] == qs[i]]
        if len(tr) < 500 or len(te) < 50: continue
        rng = np.random.default_rng(SEED)
        if len(tr) > MAX_TR:
            tr = tr.iloc[np.sort(rng.choice(len(tr), MAX_TR, replace=False))]
        if len(te) > MAX_TE:
            te = te.iloc[rng.choice(len(te), MAX_TE, replace=False)]
        A = augment_clean18(base12_leakfree(tr[BASE].values))
        Ate = augment_clean18(base12_leakfree(te[BASE].values))
        ytr, yte = tr['IV'].values, te['IV'].values
        dd = A.shape[1]; cut = int(0.8 * len(ytr)); fold = str(qs[i])

        gr = np.random.default_rng(SEED)
        grid = [GRID[k] for k in gr.choice(len(GRID), BUDGET, replace=False)]
        t0 = time.perf_counter(); best = (None, -9e9)
        for prm in grid:
            M = nblocks(A[:cut], prm['tau'])
            m = ExtraTreesRegressor(n_estimators=N_IN, max_features=M / dd,
                                    min_samples_leaf=prm['leaf'],
                                    max_depth=prm['depth'],
                                    min_samples_split=prm['split'],
                                    bootstrap=False, n_jobs=-1, random_state=SEED)
            s = r2_score(ytr[cut:], m.fit(A[:cut], ytr[:cut]).predict(A[cut:]))
            if s > best[1]: best = (prm, s)
        prm = best[0]; search_s = time.perf_counter() - t0

        for name, par in (('Dynamic ET (tuned)', prm),
                          ('Dynamic ET', dict(tau=0.75, leaf=1, depth=None, split=2))):
            M = nblocks(A, par['tau'])
            t1 = time.perf_counter()
            p = ExtraTreesRegressor(n_estimators=N_FULL, max_features=M / dd,
                                    min_samples_leaf=par['leaf'],
                                    max_depth=par['depth'],
                                    min_samples_split=par['split'],
                                    bootstrap=False, n_jobs=-1,
                                    random_state=SEED).fit(A, ytr).predict(Ate)
            rows.append(dict(fold=fold, model=name, d=dd, M_blocks=M,
                             params=str(par), r2=r2_score(yte, p),
                             mae=mean_absolute_error(yte, p),
                             search_s=search_s if 'tuned' in name else np.nan,
                             fit_s=time.perf_counter() - t1))
        print(f'{fold}  tuned {rows[-2]["r2"]:.4f}  plain {rows[-1]["r2"]:.4f}  '
              f'{prm}  (search {search_s:.0f}s)', flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)

    t = pd.DataFrame(rows)
    print(f'\n{len(t)} fits over {t.fold.nunique()} folds -> {path}\n')
    print(t.groupby('model')[['r2', 'mae', 'search_s', 'fit_s']].mean()
           .round(4).to_string())


if __name__ == '__main__':
    main()
