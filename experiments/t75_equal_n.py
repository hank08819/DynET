"""t75_equal_n.py -- 同样的训练行数下与 TabPFN 比较.

审稿意见指出, 正文里 TabPFN 只用了 10,000 行训练, 而本方法用 30,000 行, 样本量
不对等. 把本方法也压到同一个 10,000 行, 是对这一条最直接的回答, 而且不需要把
TabPFN 抬到 30,000 行(它的上下文注意力是平方级的, 那一版只在大内存机器上跑得动).

三条臂读到的行完全相同: 同一折、同一批抽出的 10,000 行训练, 同一批 15,000 行测试.
Dynamic ET 与消融读增广后的十五列, TabPFN 读面板自带的七列, 与正文一致.
"""
import os, sys, time, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from scipy.stats import wilcoxon
from t4_clean_sweep import augment, blocks, BASE
from t24_underlyings import load, PANELS, DATA

CAP = int(os.environ.get('TRAIN_CAP', '10000'))
# 森林与 TabPFN 分进程跑: sklearn 的 OpenMP 与 PyTorch 的线程在同一进程里会段错误,
# 这与 t63 里按库分组的理由相同.
WHICH = os.environ.get('ARMS', 'trees')
MAX_TE, SEED, NT = 15_000, 42, 1000
HERE = Path(__file__).resolve().parents[1]


def main(tick='AAPL'):
    d = load(DATA / PANELS[tick]); d['q'] = d['date'].dt.to_period('Q')
    qs = sorted(d['q'].unique()); rows = []
    path = HERE / 'results' / f'equal_n_{tick}_{CAP}_{WHICH}.csv'
    for i in range(4, len(qs)):
        tr = d[d['q'].isin(qs[i-4:i])]; te = d[d['q'] == qs[i]]
        if len(tr) < 500 or len(te) < 50: continue
        rng = np.random.default_rng(SEED)
        if len(tr) > 30_000: tr = tr.iloc[rng.choice(len(tr), 30_000, replace=False)]
        if len(te) > MAX_TE: te = te.iloc[rng.choice(len(te), MAX_TE, replace=False)]
        if len(tr) > CAP:                       # 三条臂共用这同一批行
            j = np.sort(np.random.default_rng(SEED).choice(len(tr), CAP, replace=False))
            tr = tr.iloc[j]
        B, Bte = tr[BASE].values, te[BASE].values
        A, Ate = augment(B), augment(Bte)
        ytr, yte = tr['IV'].values, te['IV'].values
        M, dd = blocks(A), A.shape[1]
        fold = str(qs[i]); out = []
        et = lambda mf: ExtraTreesRegressor(n_estimators=NT, max_features=mf,
                bootstrap=False, min_samples_leaf=1, n_jobs=-1, random_state=SEED)
        arms = [('Dynamic ET', A, Ate, et(M / dd)), ('ET-enr', A, Ate, et(1.0))] \
            if WHICH == 'trees' else []
        for name, X, Xt, mdl in arms:
            t0 = time.perf_counter()
            p = mdl.fit(X, ytr).predict(Xt)
            rows.append(dict(tick=tick, fold=fold, model=name, n_train=len(ytr),
                             r2=float(r2_score(yte, p)), secs=time.perf_counter()-t0))
            out.append(f'{name} {rows[-1]["r2"]:.4f}')
        try:
            if WHICH != 'tabpfn': raise RuntimeError('skip')
            from tabpfn import TabPFNRegressor
            t0 = time.perf_counter()
            m = TabPFNRegressor(device='cpu', random_state=SEED,
                                ignore_pretraining_limits=True).fit(B, ytr)
            p = m.predict(Bte)
            rows.append(dict(tick=tick, fold=fold, model='TabPFN', n_train=len(ytr),
                             r2=float(r2_score(yte, p)), secs=time.perf_counter()-t0))
            out.append(f'TabPFN {rows[-1]["r2"]:.4f}')
        except Exception as e:
            if WHICH == 'tabpfn': out.append(f'TabPFN FAIL {str(e)[:40]}')
        print(f'{fold}  n={len(ytr)}  ' + '  '.join(out), flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)

    import glob
    fr = [pd.read_csv(f) for f in glob.glob(str(HERE / 'results' / f'equal_n_{tick}_{CAP}_*.csv'))]
    t = pd.concat(fr, ignore_index=True) if fr else pd.DataFrame(rows)
    piv = t.pivot_table(index='fold', columns='model', values='r2')
    print(f'\n{t.fold.nunique()} folds at n={CAP} -> {path}\n')
    print(piv.mean().round(4).to_string())
    for m in [c for c in piv.columns if c != 'Dynamic ET']:
        dd_ = (piv['Dynamic ET'] - piv[m]).dropna()
        print(f'  vs {m:10} {dd_.mean():+.4f}  {int((dd_>0).sum())}/{len(dd_)}  '
              f'p={wilcoxon(dd_, alternative="greater").pvalue:.3g}')
    print('\n平均拟合秒数:'); print(t.groupby('model').secs.mean().round(2).to_string())


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'AAPL')
