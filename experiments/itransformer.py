"""itransformer.py -- an inverted Transformer regressor for cross-sectional data.

Liu et al. (ICLR 2024) invert the usual arrangement: a token is a variate rather
than a time step, and attention runs across variates so the model learns which
features move together. On a cross-section that reading is the natural one, since
there is no time axis within a contract, and it makes the architecture a fair
comparator for a method whose whole claim is about correlated feature blocks.

Each of the d features becomes one token, embedded from its scalar value, passed
through pre-norm self-attention and a feed-forward block, then pooled and read
out. sklearn-style fit/predict, standardised inputs, torch on CPU or MPS.
"""
import numpy as np


class ITransformerRegressor:
    def __init__(self, d_model=64, n_heads=4, n_layers=2, dropout=0.1,
                 epochs=120, lr=3e-3, batch=512, seed=42):
        self.d_model, self.n_heads, self.n_layers = d_model, n_heads, n_layers
        self.dropout, self.epochs, self.lr = dropout, epochs, lr
        self.batch, self.seed = batch, seed

    def _build(self, d):
        import torch, torch.nn as nn

        class Net(nn.Module):
            def __init__(s, d, dm, nh, nl, do):
                super().__init__()
                # per-variate embedding: variate j gets its own vector, scaled
                # by its value. A shared Linear(1, dm) would make the tokens
                # differ only by position, which is not what inversion means.
                s.W = nn.Parameter(torch.randn(1, d, dm) * (dm ** -0.5))
                s.b = nn.Parameter(torch.zeros(1, d, dm))
                layer = nn.TransformerEncoderLayer(
                    d_model=dm, nhead=nh, dim_feedforward=4 * dm, dropout=do,
                    batch_first=True, norm_first=True, activation='gelu')
                s.enc = nn.TransformerEncoder(layer, num_layers=nl)
                s.norm = nn.LayerNorm(dm)
                s.head = nn.Sequential(nn.Linear(dm, dm), nn.GELU(),
                                       nn.Linear(dm, 1))

            def forward(s, x):                            # x: (B, d)
                h = x.unsqueeze(-1) * s.W + s.b           # (B, d, dm)
                h = s.norm(s.enc(h)).mean(dim=1)          # pool over variates
                return s.head(h).squeeze(-1)

        return Net(d, self.d_model, self.n_heads, self.n_layers, self.dropout)

    def fit(self, X, y):
        import torch
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.float32)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-8
        self.ym_, self.ys_ = float(y.mean()), float(y.std() + 1e-8)
        Xs = (X - self.mu_) / self.sd_; ys = (y - self.ym_) / self.ys_
        self.dev_ = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.net_ = self._build(X.shape[1]).to(self.dev_)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.lr, total_steps=max(1, self.epochs *
                int(np.ceil(len(ys) / self.batch))))
        Xt = torch.tensor(Xs, device=self.dev_); yt = torch.tensor(ys, device=self.dev_)
        lossf = torch.nn.SmoothL1Loss()
        self.net_.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(yt), device=self.dev_)
            for i in range(0, len(yt), self.batch):
                idx = perm[i:i + self.batch]
                opt.zero_grad()
                loss = lossf(self.net_(Xt[idx]), yt[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
                opt.step(); sched.step()
        return self

    def transform(self, X):
        """The pooled variate representation, for use as features downstream."""
        import torch
        X = (np.asarray(X, dtype=np.float32) - self.mu_) / self.sd_
        self.net_.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(X), 8192):
                t = torch.tensor(X[i:i + 8192], device=self.dev_)
                h = t.unsqueeze(-1) * self.net_.W + self.net_.b
                h = self.net_.norm(self.net_.enc(h)).mean(dim=1)
                out.append(h.cpu().numpy())
        return np.concatenate(out)

    def predict(self, X):
        import torch
        X = (np.asarray(X, dtype=np.float32) - self.mu_) / self.sd_
        self.net_.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(X), 8192):
                t = torch.tensor(X[i:i + 8192], device=self.dev_)
                out.append(self.net_(t).cpu().numpy())
        return np.concatenate(out) * self.ys_ + self.ym_
