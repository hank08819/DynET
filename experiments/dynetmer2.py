"""dynetmer2.py -- DynET-mer2: the inverted encoder read per variate, not pooled.

The first version fits an inverted Transformer, pools its representation over the
variates, and appends the highest-variance coordinates of the pooled vector. The
pooling is what limits it. Inversion means a token is a variate; averaging the
tokens discards the per-feature identity, and the per-feature identity is exactly
what the block rule reads. What reaches the forest is then a set of coordinates
with no correspondence to any feature, and union-find has nothing structural to
find in them.

This version keeps the tokens. After the encoder has run its attention across
variates, variate j is a vector h_j, and the model's own readout head applied to
h_j alone is a scalar: what that variate, refined by what it attends to, would
predict on its own. There are d of them, one per input feature, and they inherit
the correlation structure of the features they came from -- so the block the rule
infers over the combined matrix is a block in the data, not an artefact of a
projection.

The forest then sees the augmented features and their attention-refined
counterparts side by side, and Step 2 infers blocks over the whole thing. A
feature and its refinement land in the same block by construction, which is the
behaviour the rule was designed for: one candidate drawn from the pair, not two.
"""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


class DynETmer2:
    def __init__(self, encoder='it', n_pool=0, tau=0.75, n_estimators=1000,
                 min_samples_leaf=1, enc_epochs=80, d_model=64, n_layers=2,
                 seed=42, augment=None):
        # 'it': the inverted encoder, which pools variates in its own forward
        # pass but whose tokens are variates throughout. 'ft': the feature
        # tokenizer of the tabular literature, which reads out through a CLS
        # token. Both carry one token per input feature, so the per-variate
        # readout below applies to either; which one serves the forest better is
        # measured, not assumed.
        self.encoder = encoder
        self.n_pool = n_pool                 # extra pooled coordinates, 0 = none
        self.tau, self.seed = tau, seed
        self.n_estimators, self.min_samples_leaf = n_estimators, min_samples_leaf
        self.enc_epochs, self.d_model, self.n_layers = enc_epochs, d_model, n_layers
        self.augment = augment               # Step 1 of Dynamic ET

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

    def _variate_scores(self, Xb):
        """One scalar per input variate: the readout head applied token by token."""
        import torch
        net = self.enc_.net_
        X = (np.asarray(Xb, dtype=np.float32) - self.enc_.mu_) / self.enc_.sd_
        net.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(X), 8192):
                t = torch.tensor(X[i:i + 8192], device=self.enc_.dev_)
                if self.encoder == 'ft':
                    h = net.norm(net.enc(net.tokens(t)))   # (B, 1+d, dm) with CLS
                    cls, h = h[:, :1], h[:, 1:]            # drop CLS for the tokens
                else:
                    h = t.unsqueeze(-1) * net.W + net.b    # (B, d, dm)
                    h = net.norm(net.enc(h))
                    cls = h.mean(dim=1, keepdim=True)
                z = net.head(h).squeeze(-1)                # (B, d), per variate
                if self.n_pool:
                    z = torch.cat([z, cls[:, 0, :self.n_pool]], dim=1)
                out.append(z.cpu().numpy())
        return np.concatenate(out)

    def _features(self, Xb):
        A = self.augment(Xb) if self.augment is not None else np.asarray(Xb)
        return np.hstack([A, self._variate_scores(Xb)])

    def fit(self, X, y):
        if self.encoder == 'ft':
            from ft_transformer import FTTransformerRegressor as Enc
        else:
            from itransformer import ITransformerRegressor as Enc
        self.enc_ = Enc(d_model=self.d_model, n_layers=self.n_layers,
                        epochs=self.enc_epochs, seed=self.seed).fit(X, y)
        Z = self._features(X)
        self.M_, self.d_ = self._blocks(Z, self.tau), Z.shape[1]
        self.et_ = ExtraTreesRegressor(
            n_estimators=self.n_estimators, max_depth=None,
            min_samples_leaf=self.min_samples_leaf, max_features=self.M_ / self.d_,
            bootstrap=False, n_jobs=-1, random_state=self.seed).fit(Z, y)
        return self

    def predict(self, X):
        return self.et_.predict(self._features(X))
