"""rotation_forest.py -- Rodriguez, Kuncheva and Alonso (2006), for regression.

Rotation Forest is the other classical answer to correlated covariates, and it
is the one this paper should be read against. Its diagnosis is the same as ours:
axis-aligned splits waste their budget when the informative directions are not
the coordinate axes. Its remedy is different. Where the block rule keeps the
axes and shrinks the candidate set to one draw per correlated group, Rotation
Forest keeps the full candidate set and changes the axes, rotating each tree's
view of the data by a PCA fitted to a random partition of the features.

Per tree: split the features into K disjoint subsets at random, draw a bootstrap
sample of the rows for each subset, fit PCA to that subset on that sample, and
assemble the components into one sparse rotation matrix. The tree is then grown
on the rotated data. Diversity comes from the rotation rather than from the
candidate draw, so every tree still considers every rotated axis.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, RegressorMixin


class RotationForestRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=300, n_features_per_subset=3,
                 sample_frac=0.75, max_depth=None, min_samples_leaf=1,
                 random_state=42):
        self.n_estimators = n_estimators
        self.n_features_per_subset = n_features_per_subset
        self.sample_frac = sample_frac
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)
        self.rot_, self.trees_ = [], []
        k = max(1, self.n_features_per_subset)
        for b in range(self.n_estimators):
            perm = rng.permutation(d)
            subsets = [perm[i:i+k] for i in range(0, d, k)]
            R = np.zeros((d, d))
            for sub in subsets:
                rows = rng.choice(n, max(k + 1, int(self.sample_frac * n)),
                                  replace=True)
                Xs = X[np.ix_(rows, sub)]
                try:
                    comp = PCA(n_components=len(sub),
                               random_state=int(rng.integers(1 << 31))).fit(Xs).components_
                except Exception:
                    comp = np.eye(len(sub))
                if comp.shape[0] < len(sub):            # degenerate subset
                    comp = np.vstack([comp, np.eye(len(sub))[comp.shape[0]:]])
                R[np.ix_(sub, sub)] = comp.T
            t = DecisionTreeRegressor(max_depth=self.max_depth,
                                      min_samples_leaf=self.min_samples_leaf,
                                      random_state=int(rng.integers(1 << 31)))
            t.fit(X @ R, y)
            self.rot_.append(R); self.trees_.append(t)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        out = np.zeros(len(X))
        for R, t in zip(self.rot_, self.trees_):
            out += t.predict(X @ R)
        return out / len(self.trees_)
