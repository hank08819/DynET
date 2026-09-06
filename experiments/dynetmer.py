"""dynetmer.py -- DynET-mer: a variate-attention encoder in front of Dynamic ET.

The block rule is a tool for correlated features. A transformer encoder produces
exactly that: a representation whose coordinates are heavily correlated by
construction, since they are pooled from attention over the same variates. So the
encoder and the rule are not two methods stacked for the sake of stacking. The
encoder manufactures the regime the rule was designed for.

The encoder is fit on the training window alone and its pooled representation is
reduced and appended to the augmented features. Blocks are then inferred over the
combined matrix, so the candidate budget accounts for both the raw features and
the learned ones. Nothing about the test quarter enters any of it.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor


class DynETmer:
    def __init__(self, n_embed=8, tau=0.75, n_estimators=1000, min_samples_leaf=1,
                 enc_epochs=80, seed=42, augment=None, whiten='pca'):
        # whiten='pca' orthogonalises the representation, which removes the very
        # correlation the block rule exists to exploit. 'none' keeps the encoder's
        # own coordinates, correlated as it produced them.
        self.whiten = whiten
        self.n_embed, self.tau = n_embed, tau
        self.n_estimators, self.min_samples_leaf = n_estimators, min_samples_leaf
        self.enc_epochs, self.seed = enc_epochs, seed
        self.augment = augment                      # the method's Step 1

    @staticmethod
    def _blocks(X, tau):
        C = np.abs(np.corrcoef(np.where(np.isfinite(X), X, 0.).T))
        np.fill_diagonal(C, 0.)
        d = C.shape[0]; par = list(range(d))
        def f(a):
            while par[a] != a: par[a] = par[par[a]]; a = par[a]
            return a
        for i in range(d):
            for j in range(i + 1, d):
                if C[i, j] > tau:
                    a, b = f(i), f(j)
                    if a != b: par[b] = a
        return len({f(i) for i in range(d)})

    def _features(self, Xb, fit=False):
        A = self.augment(Xb) if self.augment is not None else np.asarray(Xb)
        E = self.enc_.transform(Xb)
        if self.whiten == 'pca':
            E = self.pca_.fit_transform(E) if fit else self.pca_.transform(E)
        else:                                   # keep the correlated coordinates,
            if fit:                             # selecting the highest-variance ones
                self.keep_ = np.argsort(E.var(0))[::-1][:self.n_embed]
            E = E[:, self.keep_]
        return np.hstack([A, E])

    def fit(self, X, y):
        from itransformer import ITransformerRegressor
        self.enc_ = ITransformerRegressor(epochs=self.enc_epochs,
                                          seed=self.seed).fit(X, y)
        self.pca_ = PCA(n_components=self.n_embed, random_state=self.seed)
        Z = self._features(X, fit=True)
        self.M_, self.d_ = self._blocks(Z, self.tau), Z.shape[1]
        self.et_ = ExtraTreesRegressor(
            n_estimators=self.n_estimators, max_depth=None,
            min_samples_leaf=self.min_samples_leaf, max_features=self.M_ / self.d_,
            bootstrap=False, n_jobs=-1, random_state=self.seed).fit(Z, y)
        return self

    def predict(self, X):
        return self.et_.predict(self._features(X))
