"""t79_enriched_comparators.py -- 把增广后的十五列交给对手, 再比一次.

正文的对照里, 只有本方法与同特征消融读十五列增广矩阵, 其余对手读七列原始矩阵,
因为增广是本方法的第一步, 而那些方法并不含这一步. 这样比是"整条方法对现有方法",
成立; 但它无法回答"差距里有多少来自预算规则". 这一支回答后者: 同样的折、同样的
行、同样的十五列, 只换学习器.

给对手更多的列是给它们更多的信息. 如果本方法在这种设置下仍然领先, 那么"赢在特征
不在方法"这一条就被排除; 如果不领先, 主张应当写成相对同特征消融的改进.
"""
import os, sys, time, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.metrics import r2_score
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from t4_clean_sweep import augment, blocks, BASE
from t24_underlyings import load, PANELS, DATA

MAX_TR, MAX_TE, SEED, NT = 30_000, 15_000, 42, 1000
HERE = Path(__file__).resolve().parents[1]


def learners(seed=SEED):
    from catboost import CatBoostRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    return [('CatBoost', lambda: CatBoostRegressor(verbose=0, random_seed=seed)),
            ('XGBoost',  lambda: XGBRegressor(random_state=seed, n_jobs=-1)),
            ('LightGBM', lambda: LGBMRegressor(random_state=seed, n_jobs=-1, verbose=-1)),
            ('HistGBR',  lambda: HistGradientBoostingRegressor(random_state=seed))]


def main(ticks):
    rows = []
    path = HERE / 'results' / 'enriched_comparators.csv'
    for tick in ticks:
        d = load(DATA / PANELS[tick]); d['q'] = d['date'].dt.to_period('Q')
        qs = sorted(d['q'].unique())
        for i in range(4, len(qs)):
            tr = d[d['q'].isin(qs[i-4:i])]; te = d[d['q'] == qs[i]]
            if len(tr) < 500 or len(te) < 50: continue
            rng = np.random.default_rng(SEED)
            if len(tr) > MAX_TR: tr = tr.iloc[rng.choice(len(tr), MAX_TR, replace=False)]
            if len(te) > MAX_TE: te = te.iloc[rng.choice(len(te), MAX_TE, replace=False)]
            B, Bte = tr[BASE].values, te[BASE].values
            A, Ate = augment(B), augment(Bte)
            ytr, yte = tr['IV'].values, te['IV'].values
            M, dd = blocks(A), A.shape[1]
            fold = str(qs[i]); out = []

            def rec(name, cols, mk, X, Xt):
                t0 = time.perf_counter()
                p = np.asarray(mk().fit(X, ytr).predict(Xt), dtype=float)
                rows.append(dict(tick=tick, fold=fold, model=name, cols=cols,
                                 r2=float(r2_score(yte, p)),
                                 secs=time.perf_counter() - t0))
                out.append(f'{name}/{cols} {rows[-1]["r2"]:.4f}')

            et = lambda mf: (lambda: ExtraTreesRegressor(n_estimators=NT,
                    max_features=mf, bootstrap=False, min_samples_leaf=1,
                    n_jobs=-1, random_state=SEED))
            rec('Dynamic ET', 15, et(M / dd), A, Ate)
            rec('ET-enr', 15, et(1.0), A, Ate)
            for name, mk in learners():
                rec(name, 15, mk, A, Ate)     # 增广后的十五列
                rec(name, 7, mk, B, Bte)      # 原始七列, 与正文一致
            print(f'{tick} {fold}  M={M}  ' + '  '.join(out), flush=True)
            pd.DataFrame(rows).to_csv(path, index=False)
    print(f'\n{len(rows)} rows -> {path}')


if __name__ == '__main__':
    main(sys.argv[1:] or ['AAPL', 'TSLA', 'NVDA', 'SPY', 'QQQ'])
