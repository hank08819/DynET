"""t24_underlyings.py -- the same comparison on five underlyings.

Every result so far rests on AAPL 2016--2020, sixteen quarterly folds. A rule
that sets the candidate budget from the correlation structure of the training
window should not depend on which underlying wrote that window, and this is the
test of that. AAPL, NVDA, QQQ, TSLA and SPY are run under the identical
protocol -- four rolling quarters of training capped at 30,000, the next quarter
capped at 15,000, seed 42, 1,000 trees, leak-free features, Step 1 augmentation
for Dynamic ET and its ablation, base features for the comparators. The parametric surfaces, SVI and
Heston, read a contract and a maturity and are fitted per fold on the same
training window, so the traditional-finance baselines face the same task as
the learners and not an easier or harder one.

The panels differ in ways that matter: QQQ and SPY are index products with dense
strikes and tight spreads, TSLA carries a 5-for-1 split and the 2022 drawdown,
NVDA a regime of its own. If the block rule is fitted to one underlying's
idiosyncrasies it will show here.
"""
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.ensemble import (ExtraTreesRegressor, RandomForestRegressor,
                              GradientBoostingRegressor)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from scipy.stats import wilcoxon
from t4_clean_sweep import augment, blocks, BASE
from svi_baseline import SVIRegressor
from heston_baseline import HestonRegressor

DATA = Path(DATA_DIR)
PANELS = {'AAPL': 'aapl_2016_2020.csv', 'NVDA': 'nvda_2020_2022.csv',
          'QQQ': 'qqq_2020_2022.csv',  'TSLA': 'tsla_2019_2022.csv',
          'SPY': 'spy_2020_2022.csv'}
MAX_TR, MAX_TE, SEED, NT = 30_000, 15_000, 42, 1000
HERE = Path(__file__).resolve().parents[1]


def load(path):
    """t4_clean_sweep.load, parameterised by panel. Same filters throughout."""
    raw = pd.read_csv(path, low_memory=False, on_bad_lines='skip')
    raw.columns = [c.strip() for c in raw.columns]
    n = lambda c: pd.to_numeric(raw[c], errors='coerce')
    dates = pd.to_datetime(raw['[QUOTE_DATE]'], errors='coerce'); fr = []
    for t, ty in (('C', 0), ('P', 1)):
        fr.append(pd.DataFrame({'date': dates, 'S': n('[UNDERLYING_LAST]'),
            'K': n('[STRIKE]'), 'T': n('[DTE]') / 365., 'IV': n(f'[{t}_IV]'),
            'bid': n(f'[{t}_BID]'), 'ask': n(f'[{t}_ASK]'), 'is_call': 1 - ty}))
    d = pd.concat(fr, ignore_index=True).dropna()
    d = d[(d.IV > 0.01) & (d.IV < 4) & (d['T'] > 1/365) & (d.K > 0) & (d.ask >= d.bid)]
    d['m'] = d.S / d.K; d['logm'] = np.log(d.m); d['spr'] = d.ask - d.bid
    return d[d.m.between(0.5, 2.0)].sort_values('date').reset_index(drop=True)


def main():
    rows = []
    path = HERE / 'results' / 'underlyings.csv'
    for tick, fn in PANELS.items():
        f = DATA / fn
        if not f.exists():
            print(f'{tick}: {fn} not found, skipping', flush=True); continue
        d = load(f); d['q'] = d['date'].dt.to_period('Q')
        qs = sorted(d['q'].unique())
        print(f'\n===== {tick}  {len(d):,} rows  {len(qs)} quarters =====', flush=True)

        for i in range(4, len(qs)):
            tr = d[d['q'].isin(qs[i-4:i])]; te = d[d['q'] == qs[i]]
            if len(tr) < 2000 or len(te) < 500:
                continue
            rng = np.random.default_rng(SEED)
            if len(tr) > MAX_TR: tr = tr.iloc[rng.choice(len(tr), MAX_TR, replace=False)]
            if len(te) > MAX_TE: te = te.iloc[rng.choice(len(te), MAX_TE, replace=False)]
            Btr, Bte = tr[BASE].values, te[BASE].values
            Atr, Ate = augment(Btr), augment(Bte)
            ytr, yte = tr['IV'].values, te['IV'].values
            M, dd = blocks(Atr), Atr.shape[1]
            fold = str(qs[i])

            def et(mf):
                return ExtraTreesRegressor(n_estimators=NT, max_depth=None,
                                           min_samples_leaf=1, max_features=mf,
                                           bootstrap=False, n_jobs=-1,
                                           random_state=SEED)
            arms = [
                ('Dynamic ET', Atr, Ate, lambda: et(M / dd)),
                ('ET-aug',     Atr, Ate, lambda: et(1.0)),
                ('ET',         Btr, Bte, lambda: et(1.0)),
                ('RF',         Btr, Bte, lambda: RandomForestRegressor(
                    n_estimators=300, max_depth=12, min_samples_leaf=3,
                    n_jobs=-1, random_state=SEED)),
                ('GBR',        Btr, Bte, lambda: GradientBoostingRegressor(
                    n_estimators=200, max_depth=4, random_state=SEED)),
                ('XGBoost',    Btr, Bte, lambda: XGBRegressor(
                    n_estimators=600, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                    random_state=SEED, verbosity=0)),
                ('LightGBM',   Btr, Bte, lambda: LGBMRegressor(
                    n_estimators=600, num_leaves=63, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                    random_state=SEED, verbose=-1)),
                ('CatBoost',   Btr, Bte, lambda: CatBoostRegressor(
                    iterations=600, depth=6, learning_rate=0.05,
                    random_seed=SEED, verbose=0, thread_count=-1)),
                ('SVI',        Btr, Bte, lambda: SVIRegressor()),
                ('Heston',     Btr, Bte, lambda: HestonRegressor()),
            ]
            line = []
            for name, X, Xte, make in arms:
                try:
                    m = make()
                    t0 = time.perf_counter(); m.fit(X, ytr)
                    fit_s = time.perf_counter() - t0
                    p = np.asarray(m.predict(Xte), dtype=float)
                    rec = dict(ticker=tick, fold=fold, model=name, M_blocks=M, d=dd,
                               n_train=len(tr), n_test=len(te),
                               r2=r2_score(yte, p), mae=mean_absolute_error(yte, p),
                               rmse=float(np.sqrt(mean_squared_error(yte, p))),
                               fit_s=fit_s)
                    line.append(f'{name} {rec["r2"]:.4f}')
                except Exception as e:
                    rec = dict(ticker=tick, fold=fold, model=name, M_blocks=M, d=dd,
                               r2=np.nan, mae=np.nan, rmse=np.nan, fit_s=np.nan,
                               error=str(e)[:120])
                    line.append(f'{name} FAIL')
                rows.append(rec)
            print(f'  {fold}  M={M}/{dd}  ' + '  '.join(line), flush=True)
            pd.DataFrame(rows).to_csv(path, index=False)

    t = pd.DataFrame(rows)
    print(f'\n{len(t)} fits, {t.ticker.nunique()} underlyings, '
          f'{t.groupby("ticker").fold.nunique().sum()} folds -> {path}\n')
    print(t.pivot_table(index='model', columns='ticker', values='r2', aggfunc='mean')
           .assign(all=lambda x: t.groupby('model').r2.mean())
           .sort_values('all', ascending=False)
           .to_string(float_format=lambda x: f'{x:.4f}'))
    print('\n=== Dynamic ET against each comparator, all folds pooled ===')
    piv = t.pivot_table(index=['ticker', 'fold'], columns='model', values='r2')
    for m in [c for c in piv.columns if c != 'Dynamic ET']:
        dd_ = (piv['Dynamic ET'] - piv[m]).dropna()
        if len(dd_) > 2:
            p = wilcoxon(dd_, alternative='greater').pvalue
            print(f'  vs {m:<12} dR2={dd_.mean():+.4f}  ahead {int((dd_>0).sum())}'
                  f'/{len(dd_)}  p={p:.4g}')


if __name__ == '__main__':
    main()
