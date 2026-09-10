"""t66_diversity.py -- 集成分解: 树间多样性与单树损失, 逐区间.

对同一批测试行, 森林的均方误差与单棵树的平均均方误差之间有恒等式

    E_p = L_p - V_p,
    L_p = (1/B) sum_b mean_i (y_i - T_b(x_i))^2,
    V_p = (1/B) sum_b mean_i (T_b(x_i) - fbar_p(x_i))^2,

其中 fbar_p 是该森林的预测. 这是逐点配方, 不需要任何概率假设. 两个策略相减:

    E_A - E_D = (V_D - V_A) - (L_D - L_A),

于是"块预算是否值得"就化成一个可以直接测的问题: 它买到的树间多样性, 有没有
超过它在单棵树精度上付出的代价. 这一支把两项在五个区间上分别量出来.

逐棵树的预测不落盘: 只累计一阶和二阶和, 由 S1, S2 还原 fbar 与 V.
"""
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from t4_clean_sweep import load, augment, blocks, BASE
from t6_clean_otm import region_of, REGIONS

MAX_TR, MAX_TE, SEED, NT = 30_000, 15_000, 42, 1000
HERE = Path(__file__).resolve().parents[1]


def decompose(model, X, y):
    """returns per-row L (mean single-tree loss) and V (tree dispersion)."""
    S1 = np.zeros(len(X)); S2 = np.zeros(len(X))
    for t in model.estimators_:
        p = t.predict(X)
        S1 += p; S2 += p * p
    B = len(model.estimators_)
    fbar = S1 / B
    V = np.maximum(S2 / B - fbar ** 2, 0.0)
    L = y ** 2 - 2 * y * fbar + S2 / B
    return L, V, fbar


def main():
    d = load(); d['q'] = d['date'].dt.to_period('Q')
    qs = sorted(d['q'].unique()); rows = []
    path = HERE / 'results' / 'diversity.csv'
    for i in range(4, len(qs)):
        tr = d[d['q'].isin(qs[i-4:i])]; te = d[d['q'] == qs[i]]
        if len(tr) < 500 or len(te) < 50: continue
        rng = np.random.default_rng(SEED)
        if len(tr) > MAX_TR: tr = tr.iloc[rng.choice(len(tr), MAX_TR, replace=False)]
        if len(te) > MAX_TE: te = te.iloc[rng.choice(len(te), MAX_TE, replace=False)]
        A, Ate = augment(tr[BASE].values), augment(te[BASE].values)
        ytr, yte = tr['IV'].values, te['IV'].values
        M, dd = blocks(A), A.shape[1]
        reg = np.array([region_of(m, c) for m, c in
                        zip(te['m'].values, te['is_call'].values.astype(bool))])
        fold = str(qs[i]); t0 = time.perf_counter(); part = {}
        for name, mf in (('DynET', M / dd), ('all', 1.0)):
            m = ExtraTreesRegressor(n_estimators=NT, max_features=mf, bootstrap=False,
                                    min_samples_leaf=1, n_jobs=-1,
                                    random_state=SEED).fit(A, ytr)
            part[name] = decompose(m, Ate, yte)
        for r in REGIONS:
            k = reg == r
            if int(k.sum()) < 30 or np.var(yte[k]) < 1e-12: continue
            rec = dict(fold=fold, region=r, n=int(k.sum()), M=M,
                       var_y=float(np.var(yte[k])))
            for name in ('DynET', 'all'):
                L, V, f = part[name]
                rec[f'L_{name}'] = float(L[k].mean())
                rec[f'V_{name}'] = float(V[k].mean())
                rec[f'E_{name}'] = float(((f[k] - yte[k]) ** 2).mean())
            rows.append(rec)
        print(f'{fold}  M={M}/{dd}  ({time.perf_counter()-t0:.0f}s)', flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)

    t = pd.DataFrame(rows)
    g = t.groupby('region').mean(numeric_only=True).reindex(REGIONS)
    g['dV'] = g.V_DynET - g.V_all                 # 多样性买到的
    g['dL'] = g.L_DynET - g.L_all                 # 单树精度付出的
    g['dE'] = g.E_all - g.E_DynET                 # 森林误差的下降
    g['check'] = g.dV - g.dL - g.dE               # 恒等式残差
    print(f'\n{len(t)} 行, {t.fold.nunique()} 折 -> {path}\n')
    print(g[['V_all', 'V_DynET', 'dV', 'L_all', 'L_DynET', 'dL', 'dE', 'check']]
          .round(8).to_string())


if __name__ == '__main__':
    main()
