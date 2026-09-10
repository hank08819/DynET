"""t80_bf_widths.py -- Block Forests on the enriched matrix, at two widths.

The published comparison gives Block Forests the seven columns the panel
provides, because the enrichment is Step 1 of the method proposed here and is
not part of Block Forests. That answers "whole method against existing method".
It does not answer the sharper question: given the same blocked feature matrix,
does setting the budget to the block count beat the per-block sampling Block
Forests performs? This run answers that, at both enrichment widths.

Every arm uses the same tree machinery (blocked_et), so the only difference is
how a node draws its candidates. Blocks come from the correlation structure of
the matrix each arm reads. Block Forests keeps its own per-block count,
sqrt(block size), which is the ranger default; nothing is tuned for it or
against it.

scikit-learn is deliberately not imported: its OpenMP runtime and numba's
segfault in one process, which is the same reason t76 avoids it. R^2 is
computed from its definition, and the three enrichment maps are copied verbatim
from dynET.py rather than imported.
"""
import os, sys, time, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
for _m in ('bottleneck', 'numexpr'):
    sys.modules.setdefault(_m, None)
sys.path.append(str(Path(__file__).resolve().parents[1] / 'dynet-release' / 'experiments'))
import numpy as np, pandas as pd
from blocked_et import BlockedExtraTrees

MAX_TR, MAX_TE, SEED, NT, TAU = 30_000, 15_000, 42, 1000, 0.75
BASE7 = ['m', 'logm', 'T', 'K', 'S', 'is_call', 'spr']
DATA = Path(os.environ.get('OPTDATA',
            '/Users/henry_han/LHP/JMP1+MS2027/data/OptionData2026'))
PANELS = {'AAPL': 'aapl_2016_2020', 'NVDA': 'nvda_2020_2022',
          'QQQ': 'qqq_2020_2022', 'SPY': 'spy_2020_2022', 'TSLA': 'tsla_2019_2022'}
HERE = Path(__file__).resolve().parents[1]


def r2(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def augment_shape18(X):
    return augment_clean18(base12_leakfree(X))


def _bs_shapes(m, k, t, is_call, sigma=0.30):
    """The four sensitivity shapes at a fixed volatility, so nothing reads the
    target. Black-Scholes delta, gamma, vega and theta are functions of
    (S, K, tau, sigma); holding sigma at a constant leaves exactly the part
    that depends on the contract and removes the part that is the answer.
    """
    from scipy.stats import norm
    st = np.sqrt(np.maximum(t, 1e-8))
    d1 = (k + 0.5 * sigma ** 2 * t) / (sigma * st)
    d2 = d1 - sigma * st
    pdf, cdf = norm.pdf(d1), norm.cdf(d1)
    delta = np.where(is_call > 0.5, cdf, cdf - 1.0)
    gamma = pdf / (np.maximum(m, 1e-8) * sigma * st)      # per unit of S/K
    vega = pdf * st
    theta = -pdf * sigma / (2 * st)
    return delta, gamma, vega, theta

def base12_leakfree(X):
    """The twelve-column base vector, with no path to the target.

    The literature's base vector is twelve columns wide because it carries the
    vendor's delta, gamma, vega, theta and the mid quote. The width is separable
    from the leakage: the four sensitivities are functions of (S, K, tau, sigma),
    so holding sigma at a constant keeps the part that depends on the contract
    and drops the part that is the answer, and the mid quote is replaced by the
    relative spread. The twelve are then

        m, logm, T, K, S, is_call, spr,
        delta0, gamma0, vega0, theta0, spr/S

    with the four shapes evaluated at sigma = 0.30. Nothing reads the implied
    volatility or a vendor column, so Proposition 1 of the paper does not apply.
    """
    X = np.asarray(X, dtype=float)
    m, lm, t, ic, S, sp = X[:, 0], X[:, 1], X[:, 2], X[:, 5], X[:, 4], X[:, 6]
    dl, gm, vg, th = _bs_shapes(m, lm, t, ic)
    return np.column_stack([X, dl, gm, vg, th, sp / np.maximum(S, 1e-8)])

def augment_basic(X):
    """Basic augmentation, Step 1 in the leak-free regime: seven columns in,
    fifteen out.

    Eight terms are appended, in this order:

        m*T, logm^2, m*spr, spr/T, m^2, T^2, logm*T, spr*T

    Two are the smile-maturity couplings (m*T, logm*T), three are the curvature
    the smile is written in (logm^2, m^2, T^2), and three are the liquidity
    couplings a quote carries (m*spr, spr/T, spr*T). Every one is a product or
    a ratio of columns the panel already has, which is exactly why the fifteen
    span far fewer than fifteen directions and why the budget has to be counted
    in blocks.

    No sensitivity and no mid quote enters. Section IV of the paper shows the
    vendor's delta, gamma, vega and theta are algebraic functions of the target,
    recoverable from the panel's own S, K and T at correlations of 0.9996 and
    0.9947, so a model given them is given its own answer.
    """
    X = np.asarray(X, dtype=float)
    m, lm, t, sp = X[:, 0], X[:, 1], X[:, 2], X[:, 6]
    st = np.where(t > 1e-8, t, 1e-8)
    return np.hstack([X, np.column_stack([m * t, lm ** 2, m * sp, sp / st,
                                          m ** 2, t ** 2, lm * t, sp * t])])

def augment_clean18(B12):
    """Step 1 in the leak-free twelve-column regime: twelve columns in,
    eighteen out.

    Six terms are appended, in this order:

        m*T, logm^2, m*spr, spr/T, m^2, T^2

    the basic eight without the two the four sensitivity shapes already carry.
    Pass the twelve columns base12_leakfree returns.
    """
    B = np.asarray(B12, dtype=float)
    m, lm, t, sp = B[:, 0], B[:, 1], B[:, 2], B[:, 6]
    st = np.where(t > 1e-8, t, 1e-8)
    return np.hstack([B, np.column_stack([m * t, lm ** 2, m * sp, sp / st,
                                          m ** 2, t ** 2])])

def block_labels(X, tau=TAU):
    """Step 2. Union-find on |corr| > tau, returning one label per column."""
    A = np.asarray(X, dtype=float)
    C = np.abs(np.corrcoef(np.where(np.isfinite(A), A, 0.0).T))
    np.fill_diagonal(C, 0.0)
    d = C.shape[0]
    parent = list(range(d))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(d):
        for j in range(i + 1, d):
            if C[i, j] > tau:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[rb] = ra

    roots = [find(i) for i in range(d)]
    order = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return np.array([order[r] for r in roots])


def main(ticks):
    rows = []
    path = HERE / 'results' / 'bf_widths.csv'
    for tick in ticks:
        d = pd.read_parquet(DATA / (PANELS[tick] + '.parquet'))
        d['q'] = d['date'].dt.to_period('Q')
        qs = sorted(d['q'].unique())
        for i in range(4, len(qs)):
            tr = d[d['q'].isin(qs[i-4:i])]; te = d[d['q'] == qs[i]]
            if len(tr) < 500 or len(te) < 50: continue
            rng = np.random.default_rng(SEED)
            if len(tr) > MAX_TR: tr = tr.iloc[rng.choice(len(tr), MAX_TR, replace=False)]
            if len(te) > MAX_TE: te = te.iloc[rng.choice(len(te), MAX_TE, replace=False)]
            B, Bte = tr[BASE7].values, te[BASE7].values
            ytr, yte = tr['IV'].values, te['IV'].values
            fold = str(qs[i]); out = []
            for width, aug in ((15, augment_basic), (18, augment_shape18)):
                X, Xt = aug(B), aug(Bte)
                dd = X.shape[1]
                cb = block_labels(X); M = int(cb.max()) + 1
                mv = np.array([max(1, int(np.sqrt((cb == b).sum())))
                               for b in range(M)])
                arms = [('Dynamic ET',     dict(mode=0, k=M,  blocks=cb, mvec=mv)),
                        ('ET-enr',         dict(mode=0, k=dd, blocks=cb, mvec=mv)),
                        ('BF-varsel',      dict(mode=2, k=dd, blocks=cb, mvec=mv)),
                        ('BF-default',     dict(mode=4, k=dd, blocks=cb, mvec=mv)),
                        ('BF-randomblock', dict(mode=3, k=dd, blocks=cb, mvec=mv))]
                for name, kw in arms:
                    t0 = time.perf_counter()
                    p = BlockedExtraTrees(NT, random_state=SEED, **kw).fit(X, ytr).predict(Xt)
                    rows.append(dict(tick=tick, fold=fold, width=width, M=M, d=dd,
                                     model=name, r2=r2(yte, p),
                                     secs=time.perf_counter() - t0))
                    out.append(f'{name}/{width} {rows[-1]["r2"]:.4f}')
            print(f'{tick} {fold}  ' + '  '.join(out), flush=True)
            pd.DataFrame(rows).to_csv(path, index=False)
    print(f'\n{len(rows)} rows -> {path}')


if __name__ == '__main__':
    main(sys.argv[1:] or ['AAPL', 'TSLA', 'NVDA', 'SPY', 'QQQ'])
