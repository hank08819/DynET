"""t47_seeds_deep.py -- 图 12 的深度与基础模型两条: TabNet 和 TabPFN.

协议同 t42_seeds. TabPFN 没有可训练的参数, 它的随机性来自送进上下文的那一批
训练行, 所以随机状态同时控制子样本的抽取和模型自身的 random_state; 这正是要看
的量: 同一份数据换一个抽取, 它的结果会移动多少.

单独一个进程运行: 这两个模型都依赖 torch, 与三个梯度提升库各自带来的 libomp
放在同一个进程里会互相锁死.
"""
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from t4_clean_sweep import load, BASE

MAX_TR, MAX_TE, DATA_SEED = 30_000, 15_000, 42
SEEDS = [42, 7, 101, 2024, 13]
HERE = Path(__file__).resolve().parents[1]


def main():
    from tabpfn import TabPFNRegressor
    from pytorch_tabnet.tab_model import TabNetRegressor
    d = load(); d['q'] = d['date'].dt.to_period('Q')
    qs = sorted(d['q'].unique()); rows = []
    path = HERE / 'results' / 'seeds_deep.csv'

    for i in range(4, len(qs)):
        tr = d[d['q'].isin(qs[i-4:i])]; te = d[d['q'] == qs[i]]
        if len(tr) < 500 or len(te) < 50: continue
        rng = np.random.default_rng(DATA_SEED)
        if len(tr) > MAX_TR: tr = tr.iloc[rng.choice(len(tr), MAX_TR, replace=False)]
        if len(te) > MAX_TE: te = te.iloc[rng.choice(len(te), MAX_TE, replace=False)]
        Xtr, Xte = tr[BASE].values, te[BASE].values
        ytr, yte = tr['IV'].values, te['IV'].values
        fold = str(qs[i])

        for s in SEEDS:
            def tabnet():
                sx = StandardScaler().fit(Xtr)
                sy = StandardScaler().fit(ytr.reshape(-1, 1))
                m = TabNetRegressor(seed=s, verbose=0)
                m.fit(sx.transform(Xtr), sy.transform(ytr.reshape(-1, 1)),
                      max_epochs=100, patience=15, batch_size=1024,
                      virtual_batch_size=256)
                return sy.inverse_transform(m.predict(sx.transform(Xte))).ravel()

            def tabpfn():
                k = min(10_000, len(ytr))
                j = np.sort(np.random.default_rng(s).choice(len(ytr), k,
                                                            replace=False))
                return TabPFNRegressor(device='cpu', random_state=s,
                                       ignore_pretraining_limits=True
                                       ).fit(Xtr[j], ytr[j]).predict(Xte)

            line = []
            for name, fn in (('TabNet', tabnet), ('TabPFN', tabpfn)):
                t0 = time.perf_counter()
                try:
                    p = np.asarray(fn(), dtype=float)
                    rows.append(dict(fold=fold, seed=s, model=name, d=Xtr.shape[1],
                                     r2=r2_score(yte, p),
                                     mae=mean_absolute_error(yte, p),
                                     secs=time.perf_counter()-t0))
                    line.append(f'{name} {rows[-1]["r2"]:.4f}')
                except Exception as e:
                    rows.append(dict(fold=fold, seed=s, model=name, r2=np.nan,
                                     error=str(e)[:120]))
                    line.append(f'{name} FAIL')
            print(f'{fold} seed={s:<5} ' + '  '.join(line), flush=True)
            pd.DataFrame(rows).to_csv(path, index=False)

    t = pd.DataFrame(rows).dropna(subset=['r2'])
    print(f'\n{len(t)} fits -> {path}\n')
    print(t.pivot_table(index='model', columns='seed', values='r2').round(4).to_string())


if __name__ == '__main__':
    main()
