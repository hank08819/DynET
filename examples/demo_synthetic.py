"""demo_synthetic.py -- what the rule does, on synthetic data, no licensed panel.

The demo shows the two things that can be shown without an option panel: that
union-find recovers the block structure a matrix was built with, and that the
budget which follows from it costs less to fit than drawing every candidate.

Accuracy is deliberately not compared here. The rule is written for windows that
are deployed forward, where the leaves under-cover the test distribution; under a
random in-distribution split of synthetic data the candidate-count curve is
monotone in the count and the comparison says nothing about the method. Section
IX of the paper states that boundary, and the numbers that carry the claim are
the 48 quarterly walk-forward windows of Section VII, which need the licensed
panel described in README.md.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from dynet import block_labels, block_count, DynamicExtraTrees

SEED = 0
PROFILE = (4, 3, 3, 1, 1, 1, 1, 1)      # the block sizes the AAPL panel produces


def make_panel(n=30_000, profile=PROFILE, noise=0.05, rng=None):
    """One latent direction per block; a block's columns are noisy copies of it."""
    rng = rng or np.random.default_rng(SEED)
    z = rng.normal(size=(n, len(profile)))
    cols = [z[:, [j]] * rng.uniform(0.8, 1.2) + noise * rng.normal(size=(n, 1))
            for j, s in enumerate(profile) for _ in range(s)]
    X = np.hstack(cols)
    y = (np.sin(z[:, 0]) + 0.7 * z[:, 1] * z[:, 2] + 0.4 * z[:, 3] ** 2
         + 0.2 * rng.normal(size=n))
    return X, y


def main():
    rng = np.random.default_rng(SEED)
    X, y = make_panel(rng=rng)
    d = X.shape[1]

    lab = block_labels(X)
    found = tuple(sorted((int((lab == b).sum()) for b in range(lab.max() + 1)),
                         reverse=True))
    print(f'built with block sizes {PROFILE}, {d} columns')
    print(f'union-find at tau = 0.75 recovers  {found},  M = {block_count(X)}')

    print('\nfit time, 300 trees, 30,000 rows:')
    for name, mf in (('all d candidates', 1.0),
                     (f'M of d = {block_count(X)}/{d}', block_count(X) / d)):
        t0 = time.perf_counter()
        ExtraTreesRegressor(n_estimators=300, max_features=mf, bootstrap=False,
                            min_samples_leaf=1, n_jobs=-1,
                            random_state=SEED).fit(X, y)
        print(f'  {name:>22}   {time.perf_counter() - t0:5.2f} s')

    est = DynamicExtraTrees(n_estimators=300, augment_features=False,
                            random_state=SEED).fit(X, y)
    print(f'\nDynamicExtraTrees on a matrix already in its final form: '
          f'M_ = {est.M_}, d_ = {est.d_}, max_features_ = {est.max_features_:.3f}')
    print('On the seven contract-and-quote columns of an option panel, call it '
          'with augment_features=True to run Step 1 first.')


if __name__ == '__main__':
    main()
