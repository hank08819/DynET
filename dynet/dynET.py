"""dynET.py -- Dynamic Extra Trees, the reference implementation.

The method is three steps and no search.

  Step 1  ENRICHMENT.  Append terms built from the columns the panel already
          carries.  Two enrichments are defined here and both are used in the
          paper; they differ only in what the base vector is allowed to hold.
  Step 2  BLOCKS.      Join two columns whenever their absolute correlation on
          the training window exceeds tau, and count the connected components
          with union-find.  That count is M, the number of directions the
          window can tell apart.
  Step 3  BUDGET.      Draw M candidates of the d at every split node, which is
          a candidate fraction of M/d.

The trees are scikit-learn's ExtraTreesRegressor at its own defaults for depth
and leaf size, so the rule is two lines on top of a standard estimator and needs
no modification to the library.


THE TWO ENRICHMENTS
-------------------

(1) BASIC FEATURE ENRICHMENT              augment_basic:  7 columns -> 15

    base    m = S/K, k = log m, tau, K, S, is_call, spr = ask - bid
    append  m*tau, k*tau                      the smile-maturity couplings
            k^2, m^2, tau^2                   the curvature of the smile
            m*spr, spr/tau, spr*tau           the liquidity couplings

    No sensitivity and no mid quote enters at any point.  Every number in the
    main paper is computed on this vector.

(2) GREEK LEAK-FREE ENRICHMENT   base12_leakfree + augment_clean18:  12 -> 18

    The twelve-column width of the earlier literature, with the leakage taken
    out.  The base is built in three steps:

      a) replace each vendor sensitivity by the same Black-Scholes function
         evaluated at a CONSTANT volatility sigma_0 = 0.30, which keeps the part
         that depends on the contract and drops the part that is the answer;
      b) divide out the spot scale, leaving each shape a dimensionless function
         of (m, tau) and the call indicator;
      c) replace the mid quote by the relative spread spr/S, the mid being the
         price the implied volatility was solved from.

    base    the seven above, plus delta0, gamma0, vega0, theta0, spr/S
    append  m*tau, k^2, m*spr, spr/tau, m^2, tau^2

    The basic eight minus two the base already carries: k*tau is the leading
    term of d1 inside the shapes, and spr*tau is close to collinear with spr/S
    once the spot scale is divided out.

    Applying sigma = vega / (S^2 tau gamma) to the shapes returns sigma_0 and
    nothing else, so no function of them recovers the contract's own volatility.

A third function, augment_full, reproduces the twelve-feature vector of the
earlier literature WITH the vendor sensitivities.  It is provided so that regime
can be reproduced; the paper's claims are not computed on it.


USAGE
-----

    est = DynamicExtraTrees().fit(X_train, y_train)      # X: the seven columns
    y_hat = est.predict(X_test)
    est.M_, est.d_, est.max_features_                    # 8, 15, 0.533

    DynamicExtraTrees(regime='clean18')     # the Greek leak-free enrichment
    DynamicExtraTrees(augment_features=False)   # a matrix already in final form
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.utils.validation import check_is_fitted

BASE7 = ['m', 'logm', 'T', 'K', 'S', 'is_call', 'spr']
BASE12 = ['m', 'logm', 'T', 'K', 'S', 'is_call',
          'delta', 'gamma', 'vega', 'theta', 'mid', 'spr']
BASE = BASE7                      # the regime every number in the paper uses
TAU = 0.75


# =============================================================================
#  ENRICHMENT 1:  basic feature enrichment, seven columns to fifteen
# =============================================================================

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



# =============================================================================
#  ENRICHMENT 2:  Greek leak-free enrichment, twelve columns to eighteen
# =============================================================================

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


def augment_clean20(B12):
    """The leak-free twelve-column regime at the full eight terms: twelve in,
    twenty out. Provided for symmetry with the twenty-column leaky regime; the
    paper reports the eighteen-column form, since the two terms this adds sit
    inside blocks the eighteen already has.
    """
    B = np.asarray(B12, dtype=float)
    m, lm, t, sp = B[:, 0], B[:, 1], B[:, 2], B[:, 6]
    st = np.where(t > 1e-8, t, 1e-8)
    return np.hstack([B, np.column_stack([m * t, lm ** 2, m * sp, sp / st,
                                          m ** 2, t ** 2, lm * t, sp * t])])


def augment_shape18(X):
    """Convenience: seven contract-and-quote columns straight to eighteen.

    Equivalent to augment_clean18(base12_leakfree(X)), for callers that hold
    only the seven base columns.
    """
    return augment_clean18(base12_leakfree(X))


augment = augment_basic          # the name the rest of the code base uses


# =============================================================================
#  The twelve-feature regime of the earlier literature, with the leakage in it
# =============================================================================

def augment_full(X):
    """Full augmentation, the twelve-feature regime: twelve columns in, twenty
    out, and the one the paper does not use.

    This is the feature vector the earlier literature uses, the contract and the
    quote plus the vendor's delta, gamma, vega, theta and the mid. The same
    eight slots are filled, with a sensitivity standing in wherever one is
    available:

        m*T, logm^2, delta*spr, spr/vega, vega*m, T^2, delta^2, vega/T

    It is provided so the twenty-column regime can be reproduced (Section S4 of
    the supplement reports it), not because the paper's claims rest on it. Every
    number in the paper is computed on the fifteen leak-free columns instead.
    """
    X = np.asarray(X, dtype=float)
    m, lm, t = X[:, 0], X[:, 1], X[:, 2]
    dl, vg, sp = X[:, 6], X[:, 8], X[:, 11]
    sv = np.where(np.abs(vg) > 1e-8, vg, 1e-8)
    st = np.where(t > 1e-8, t, 1e-8)
    return np.hstack([X, np.column_stack([m * t, lm ** 2, dl * sp, sp / sv,
                                          vg * m, t ** 2, dl ** 2, vg / st])])


# =============================================================================
#  Steps 2 and 3:  blocks by union-find, and the budget that follows
# =============================================================================

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


def block_count(X, tau=TAU):
    """Step 2, the number the rule needs: how many blocks the window has."""
    return int(block_labels(X, tau).max()) + 1


class DynamicExtraTrees(BaseEstimator, RegressorMixin):
    """Extra Trees whose candidate budget is read off the training window.

    Parameters
    ----------
    n_estimators : int
        Trees in the ensemble. The paper uses 1000 throughout.
    tau : float
        Correlation threshold that defines a block. Fixed at 0.75 for every
        panel and every fold in the paper; Section S6 of the supplement reports
        what happens on either side of it.
    augment_features : bool
        Apply Step 1 before inferring the blocks. Set False when the matrix
        already carries the columns the model should see.
    regime : {'clean15', 'clean18', 'full20'}
        Which Step 1 to apply. 'clean15' is the leak-free vector the paper's
        numbers are computed on, seven columns to fifteen. 'full20' is the
        twelve-feature vector of the earlier literature, twelve columns to
        twenty, which carries the vendor sensitivities.
    random_state, n_jobs, min_samples_leaf, max_depth, min_samples_split
        Passed to ExtraTreesRegressor unchanged.

    Attributes
    ----------
    M_ : int
        Blocks the training window produced.
    d_ : int
        Columns after Step 1.
    max_features_ : float
        The budget actually used, M_/d_.
    """

    def __init__(self, n_estimators=1000, tau=TAU, augment_features=True,
                 regime='clean15', min_samples_leaf=1, max_depth=None,
                 min_samples_split=2, random_state=42, n_jobs=-1):
        self.n_estimators = n_estimators
        self.tau = tau
        self.augment_features = augment_features
        self.regime = regime
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _matrix(self, X):
        if not self.augment_features:
            return np.asarray(X, dtype=float)
        return {'full20': augment_full,
                'clean18': augment_shape18}.get(self.regime, augment_basic)(X)

    def fit(self, X, y):
        A = self._matrix(X)
        self.d_ = A.shape[1]
        self.M_ = block_count(A, self.tau)                     # Step 2
        self.max_features_ = self.M_ / self.d_                 # Step 3
        self.forest_ = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            max_features=self.max_features_,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            bootstrap=False,
            random_state=self.random_state,
            n_jobs=self.n_jobs).fit(A, np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        check_is_fitted(self, 'forest_')
        return self.forest_.predict(self._matrix(X))


if __name__ == '__main__':
    # One quarterly fold of the AAPL panel, the protocol of Section V-B:
    # four rolling quarters of training against the quarter that follows.
    import sys, warnings
    from pathlib import Path
    warnings.filterwarnings('ignore')
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pandas as pd
    from sklearn.metrics import r2_score
    from t4_clean_sweep import load

    d = load()
    d['q'] = d['date'].dt.to_period('Q')
    qs = sorted(d['q'].unique())
    tr = d[d['q'].isin(qs[4:8])]
    te = d[d['q'] == qs[8]]
    rng = np.random.default_rng(42)
    tr = tr.iloc[rng.choice(len(tr), min(30_000, len(tr)), replace=False)]
    te = te.iloc[rng.choice(len(te), min(15_000, len(te)), replace=False)]

    est = DynamicExtraTrees().fit(tr[BASE7].values, tr['IV'].values)
    r2 = r2_score(te['IV'].values, est.predict(te[BASE7].values))
    print(f'fold {qs[8]}   M = {est.M_}/{est.d_}   '
          f'candidate fraction {est.max_features_:.3f}   R2 = {r2:.4f}')
