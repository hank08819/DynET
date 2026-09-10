"""t63_wing_models.py -- 两翼上的逐模型比较.

深度虚值和深度实值是曲面上最难的两段, 也是金融上最值得预测的两段: 深度虚值合约
的报价几乎全部是波动率信息, 尾部风险和保护性头寸都定在那里. 这一支把整张比较表
按区间重做, 在每一折上分别给出五个区间的 R^2 和 Vega 加权的价格均方误差, 然后在
两翼上对 Dynamic ET 与每一个对手做配对检验, 并在对手集合上做 Holm 校正.

各臂读到的信息与正文一致: 特征增广是 Dynamic ET 自己的第一步, 随它一起走;
其余各臂读七列准入特征. Vega 只用于加权评价, 不进入任何模型的输入.

按库分进程跑, 避免 numba 与多个 OpenMP 运行时同处一进程.
    python t63_wing_models.py core     # 树模型, MLP, SVI, Heston
    python t63_wing_models.py boost    # CatBoost, XGBoost, LightGBM, HistGBR
    python t63_wing_models.py tabpfn   # TabPFN
"""
import os, sys, time, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, '/Users/henry_han/LHP/JMP2+OptionAgent/code')
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.metrics import r2_score
from t4_clean_sweep import augment, blocks, BASE
from t6_clean_otm import region_of, REGIONS
from t24_underlyings import load, PANELS, DATA

MAX_TR, MAX_TE, SEED, NT = 30_000, 15_000, 42, 1000
# 行抽样的种子与森林的种子分开: 协议固定在 42, 但审稿意见要求检验
# 结论对"抽哪些行"是否敏感, 所以这一个可以由环境变量另行指定.
DATA_SEED = int(os.environ.get('DATA_SEED', SEED))
HERE = Path(__file__).resolve().parents[1]


def bs_vega(S, K, T, sig):
    st = np.sqrt(np.maximum(T, 1e-12))
    d1 = (np.log(np.maximum(S, 1e-12) / np.maximum(K, 1e-12)) + 0.5 * sig ** 2 * T) \
        / np.maximum(sig * st, 1e-12)
    return S * norm.pdf(d1) * st


def arms_core(A, B, M, dd):
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from svi_baseline import SVIRegressor
    from heston_baseline import HestonRegressor
    et = lambda mf: ExtraTreesRegressor(n_estimators=NT, max_depth=None,
            min_samples_leaf=1, max_features=mf, bootstrap=False,
            n_jobs=-1, random_state=SEED)
    return [('Dynamic ET', A, et(M / dd)),
            ('ET-enr',     A, et(1.0)),
            ('ET',         B, et(1.0)),
            ('RF',         B, RandomForestRegressor(n_estimators=300, max_depth=12,
                              min_samples_leaf=3, n_jobs=-1, random_state=SEED)),
            ('MLP',        B, MLPRegressor(hidden_layer_sizes=(64, 32),
                              max_iter=300, random_state=SEED)),
            ('SVI',        B, SVIRegressor()),
            ('Heston',     B, HestonRegressor())]


def arms_abl(A, B, M, dd):
    """the ablation on its own: the budget rule against drawing every candidate."""
    from sklearn.ensemble import ExtraTreesRegressor
    et = lambda mf: ExtraTreesRegressor(n_estimators=NT, max_features=mf,
            bootstrap=False, min_samples_leaf=1, n_jobs=-1, random_state=SEED)
    return [('Dynamic ET', A, et(M / dd)), ('ET-enr', A, et(1.0))]


def arms_rot(A, B, M, dd):
    from rotation_forest import RotationForestRegressor
    return [('Rotation Forest', B, RotationForestRegressor(n_estimators=300,
             n_features_per_subset=3, max_depth=12, random_state=SEED))]


def arms_boost(A, B, M, dd):
    from catboost import CatBoostRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    return [('CatBoost', B, CatBoostRegressor(verbose=0, random_seed=SEED)),
            ('XGBoost',  B, XGBRegressor(random_state=SEED, n_jobs=-1)),
            ('LightGBM', B, LGBMRegressor(random_state=SEED, n_jobs=-1, verbose=-1)),
            ('HistGBR',  B, HistGradientBoostingRegressor(random_state=SEED))]


class SeqAdapter:
    """TorchSeqRegressor indexes X with .values; give it a frame either way."""
    def __init__(self, kind):
        from optionetagent.baselines import TorchSeqRegressor
        self.m = TorchSeqRegressor(kind=kind, seed=SEED)

    def fit(self, X, y):
        self.m.fit(pd.DataFrame(np.asarray(X)), np.asarray(y)); return self

    def predict(self, X):
        return self.m.predict(pd.DataFrame(np.asarray(X)))


def arms_seq(A, B, M, dd):
    from itransformer import ITransformerRegressor
    return [('LSTM', B, SeqAdapter('lstm')),
            ('GRU',  B, SeqAdapter('gru')),
            ('iTransformer', B, ITransformerRegressor(seed=SEED))]


class TabPFN:
    # 训练行上限可由环境变量给出: 默认 10,000 是该模型的常用设置, 审稿意见要求
    # 在与本方法相同的 30,000 行上再跑一次, 以排除样本量不对等.
    CAP = int(os.environ.get('TABPFN_N', '10000'))

    def fit(self, X, y):
        from tabpfn import TabPFNRegressor
        k = min(self.CAP, len(y))
        j = np.sort(np.random.default_rng(SEED).choice(len(y), k, replace=False))
        self.m = TabPFNRegressor(device='cpu', random_state=SEED,
                                 ignore_pretraining_limits=True).fit(X[j], y[j])
        return self

    def predict(self, X):
        return self.m.predict(X)


def main(which, tick='AAPL'):
    build = {'core': arms_core, 'boost': arms_boost, 'rot': arms_rot, 'seq': arms_seq,
             'abl': arms_abl,
             'tabpfn': lambda A, B, M, dd: [('TabPFN', B, TabPFN())]}[which]
    d = load(DATA / PANELS[tick]); d['q'] = d['date'].dt.to_period('Q')
    qs = sorted(d['q'].unique()); rows = []
    tag = '' if DATA_SEED == SEED else f'_ds{DATA_SEED}'
    if which == 'tabpfn' and TabPFN.CAP != 10_000: tag += f'_n{TabPFN.CAP}'
    path = HERE / 'results' / f'wing_models_{tick}_{which}{tag}.csv'
    for i in range(4, len(qs)):
        tr = d[d['q'].isin(qs[i-4:i])]; te = d[d['q'] == qs[i]]
        if len(tr) < 500 or len(te) < 50: continue
        rng = np.random.default_rng(DATA_SEED)
        if len(tr) > MAX_TR: tr = tr.iloc[rng.choice(len(tr), MAX_TR, replace=False)]
        if len(te) > MAX_TE: te = te.iloc[rng.choice(len(te), MAX_TE, replace=False)]
        B, Bte = tr[BASE].values, te[BASE].values
        A, Ate = augment(B), augment(Bte)
        ytr, yte = tr['IV'].values, te['IV'].values
        M, dd = blocks(A), A.shape[1]
        reg = np.array([region_of(m, c) for m, c in
                        zip(te['m'].values, te['is_call'].values.astype(bool))])
        nu = bs_vega(te['S'].values, te['K'].values, te['T'].values, yte)
        fold = str(qs[i]); out = []; keep = {}
        for name, Xtr, mdl in build(A, B, M, dd):
            Xte = Ate if Xtr is A else Bte
            t0 = time.perf_counter()
            try:
                p = np.asarray(mdl.fit(Xtr, ytr).predict(Xte), dtype=float)
                if not np.isfinite(p).all(): p = np.nan_to_num(p, nan=float(ytr.mean()))
            except Exception as e:
                out.append(f'{name} FAIL {str(e)[:40]}'); continue
            secs = time.perf_counter() - t0
            keep[name] = p.astype(np.float32)
            for r in REGIONS:
                k = reg == r
                if int(k.sum()) < 30 or np.var(yte[k]) < 1e-12: continue
                e = p[k] - yte[k]
                rows.append(dict(tick=tick, fold=fold, model=name, region=r, n=int(k.sum()),
                                 var_y=float(np.var(yte[k])),
                                 r2=float(r2_score(yte[k], p[k])),
                                 mse_iv=float((e ** 2).mean()),
                                 mae_iv=float(np.abs(e).mean()),
                                 medae_iv=float(np.median(np.abs(e))),
                                 mse_px=float(((nu[k] * e) ** 2).mean()),
                                 mae_px=float(np.abs(nu[k] * e).mean()), secs=secs))
            out.append(f'{name} {r2_score(yte, p):.4f}')
        # 逐行预测存下来, 以后要换任何一个误差口径都不必重新拟合
        pdir = HERE / 'results' / 'preds'; pdir.mkdir(exist_ok=True)
        np.savez_compressed(pdir / f'{tick}_{fold}_{which}{tag}.npz',
                            y=yte.astype(np.float32), vega=nu.astype(np.float32),
                            region=reg.astype('U8'), **keep)
        print(f'{fold}  ' + '  '.join(out), flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)
    print(f'\n{len(rows)} rows -> {path}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'core',
         sys.argv[2] if len(sys.argv) > 2 else 'AAPL')
