"""t6_clean_otm.py -- the regional question again, without the leaking features.

t2 asked whether the block rule earns its keep on out-of-the-money contracts and
answered no. It ran on the twelve features that include the Greeks, where by t3
and t5 the target is already in the input, so the answer was a property of the
setup rather than of the rule. This repeats it on the leak-free feature set at
the clean protocol. Theorem 2 predicts the advantage concentrates on OTM, and
this is the first setting in which that prediction can be tested at all.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from t4_clean_sweep import load, augment, blocks, et, BASE, N_TR, N_TE, SEED

REGIONS = ['deep_OTM', 'OTM', 'ATM', 'ITM', 'deep_ITM']


def region_of(m, is_call):                      # controls' otm_analysis.py, verbatim
    if is_call:
        if m < 0.85:  return 'deep_OTM'
        if m < 0.97:  return 'OTM'
        if m <= 1.03: return 'ATM'
        if m <= 1.15: return 'ITM'
        return 'deep_ITM'
    if m > 1.15:  return 'deep_OTM'
    if m > 1.03:  return 'OTM'
    if m >= 0.97: return 'ATM'
    if m >= 0.85: return 'ITM'
    return 'deep_ITM'


def main():
    d = load(); d['q'] = d['date'].dt.to_period('Q')
    qs = sorted(d['q'].unique()); rows = []
    path = Path(__file__).resolve().parents[1] / 'results' / 'clean_otm.csv'
    for i in range(4, len(qs)):
        tr = d[d['q'].isin(qs[i-4:i])]; te = d[d['q'] == qs[i]]
        if len(tr) < 500 or len(te) < 50:
            continue
        rng = np.random.default_rng(SEED)
        if len(tr) > N_TR: tr = tr.iloc[rng.choice(len(tr), N_TR, replace=False)]
        if len(te) > N_TE: te = te.iloc[rng.choice(len(te), N_TE, replace=False)]
        B_tr, B_te = tr[BASE].values, te[BASE].values
        A_tr, A_te = augment(B_tr), augment(B_te)
        ytr, yte = tr['IV'].values, te['IV'].values
        M, dd = blocks(A_tr), A_tr.shape[1]
        reg = np.array([region_of(m, c) for m, c in
                        zip(te['m'].values, te['is_call'].values.astype(bool))])
        fold = str(qs[i])
        print(f"\nFold {fold}  test={len(te):,}  M={M}/{dd}  "
              + '  '.join(f'{r}={int((reg==r).sum())}' for r in REGIONS), flush=True)
        preds = {}
        for name, A, Bt, mf in [('base-7', B_tr, B_te, 1.0),
                                ('aug@1.00', A_tr, A_te, 1.0),
                                ('DynET', A_tr, A_te, M / dd)]:
            t0 = time.perf_counter()
            preds[name] = et(mf).fit(A, ytr).predict(Bt)
            print(f"  {name:<9} mf={mf:.3f}  overall R2="
                  f"{r2_score(yte, preds[name]):.4f}  ({time.perf_counter()-t0:.0f}s)",
                  flush=True)
        for r in REGIONS:
            k = reg == r
            if int(k.sum()) < 30 or np.var(yte[k]) < 1e-12:
                continue
            for name, p in preds.items():
                rows.append(dict(fold=fold, region=r, arm=name, n=int(k.sum()),
                                 M_blocks=M, r2=r2_score(yte[k], p[k]),
                                 mae=mean_absolute_error(yte[k], p[k])))
        pd.DataFrame(rows).to_csv(path, index=False)
    t = pd.DataFrame(rows)
    print(f'\n{len(t)} rows over {t.fold.nunique()} folds -> {path}')


if __name__ == '__main__':
    main()
