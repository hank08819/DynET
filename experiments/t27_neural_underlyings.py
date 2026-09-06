"""t27_neural_underlyings.py -- the neural comparators on the same 48 folds.

t24 measured the tree ensembles and the parametric surfaces on five underlyings.
This adds the neural comparators on the identical folds, the identical rows and
the identical seed, so the two files merge into one table. Every arm here reads
the same seven base features the other comparators read.
"""
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from t4_clean_sweep import augment, blocks, BASE

# The panel loader is repeated here rather than imported from t24. Importing that
# module pulls in XGBoost, LightGBM and CatBoost, each of which loads its own
# OpenMP runtime; torch's recurrent kernels then abort the process without
# raising. Keeping this file free of those imports keeps the arms runnable.
DATA = Path(DATA_DIR)
PANELS = {'AAPL': 'aapl_2016_2020.csv', 'NVDA': 'nvda_2020_2022.csv',
          'QQQ': 'qqq_2020_2022.csv',  'TSLA': 'tsla_2019_2022.csv',
          'SPY': 'spy_2020_2022.csv'}
MAX_TR, MAX_TE, SEED = 30_000, 15_000, 42


def load(path):
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
from itransformer import ITransformerRegressor
from ft_transformer import FTTransformerRegressor
from seq_baseline import SeqRegressor

HERE = Path(__file__).resolve().parents[1]


def main():
    rows = []
    path = HERE / 'results' / 'underlyings_neural.csv'
    for tick, fn in PANELS.items():
        f = DATA / fn
        if not f.exists():
            continue
        d = load(f); d['q'] = d['date'].dt.to_period('Q')
        qs = sorted(d['q'].unique())
        print(f'\n===== {tick}  {len(d):,} rows =====', flush=True)
        for i in range(4, len(qs)):
            tr = d[d['q'].isin(qs[i-4:i])]; te = d[d['q'] == qs[i]]
            if len(tr) < 2000 or len(te) < 500:
                continue
            rng = np.random.default_rng(SEED)
            if len(tr) > MAX_TR: tr = tr.iloc[rng.choice(len(tr), MAX_TR, replace=False)]
            if len(te) > MAX_TE: te = te.iloc[rng.choice(len(te), MAX_TE, replace=False)]
            Btr, Bte = tr[BASE].values, te[BASE].values
            ytr, yte = tr['IV'].values, te['IV'].values
            M, dd = blocks(augment(Btr)), 15
            fold = str(qs[i])
            arms = [
                ('MLP',            lambda: make_pipeline(StandardScaler(),
                    MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=400,
                                 early_stopping=True, random_state=SEED))),
                ('LSTM',           lambda: SeqRegressor(kind='lstm', seed=SEED)),
                ('GRU',            lambda: SeqRegressor(kind='gru', seed=SEED)),
                ('iTransformer',   lambda: ITransformerRegressor(seed=SEED)),
                ('FT-Transformer', lambda: FTTransformerRegressor(seed=SEED)),
            ]
            line = []
            for name, make in arms:
                try:
                    m = make()
                    t0 = time.perf_counter(); m.fit(Btr, ytr)
                    fit_s = time.perf_counter() - t0
                    p = np.asarray(m.predict(Bte), dtype=float)
                    ok = np.isfinite(p)
                    if ok.sum() < len(p) * 0.5:
                        raise ValueError('non-finite predictions')
                    p = np.where(ok, p, np.nanmedian(p[ok]))
                    rec = dict(ticker=tick, fold=fold, model=name, M_blocks=M, d=7,
                               n_train=len(tr), n_test=len(te),
                               r2=r2_score(yte, p), mae=mean_absolute_error(yte, p),
                               rmse=float(np.sqrt(mean_squared_error(yte, p))),
                               fit_s=fit_s)
                    line.append(f'{name} {rec["r2"]:.4f}')
                except Exception as e:
                    rec = dict(ticker=tick, fold=fold, model=name, M_blocks=M, d=7,
                               r2=np.nan, mae=np.nan, rmse=np.nan, fit_s=np.nan,
                               error=str(e)[:120])
                    line.append(f'{name} FAIL')
                rows.append(rec)
            print(f'  {fold}  ' + '  '.join(line), flush=True)
            pd.DataFrame(rows).to_csv(path, index=False)

    t = pd.DataFrame(rows)
    print(f'\n{len(t)} fits -> {path}\n')
    print(t.pivot_table(index='model', columns='ticker', values='r2', aggfunc='mean')
           .assign(all=lambda x: t.groupby('model').r2.mean())
           .sort_values('all', ascending=False)
           .to_string(float_format=lambda x: f'{x:.4f}'))


if __name__ == '__main__':
    main()
