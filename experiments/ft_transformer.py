"""ft_transformer.py -- FT-Transformer (Gorishniy et al., NeurIPS 2021).

A feature tokenizer maps every column to its own vector, a [CLS] token is
prepended, a pre-norm transformer runs over the resulting sequence, and the
prediction is read from CLS. It differs from the inverted encoder used elsewhere
here mainly in the readout: CLS attends to the variates rather than pooling them,
which is the arrangement the tabular literature reports as the stronger of the
two. Included so the comparison is against the architecture the field actually
uses, not a variant of our own.
"""
import numpy as np


class FTTransformerRegressor:
    def __init__(self, d_model=64, n_heads=8, n_layers=3, dropout=0.1,
                 epochs=120, lr=1e-3, batch=512, seed=42):
        self.d_model, self.n_heads, self.n_layers = d_model, n_heads, n_layers
        self.dropout, self.epochs, self.lr = dropout, epochs, lr
        self.batch, self.seed = batch, seed

    def _build(self, d):
        import torch, torch.nn as nn

        class Net(nn.Module):
            def __init__(s, d, dm, nh, nl, do):
                super().__init__()
                s.W = nn.Parameter(torch.randn(1, d, dm) * (dm ** -0.5))
                s.b = nn.Parameter(torch.zeros(1, d, dm))
                s.cls = nn.Parameter(torch.randn(1, 1, dm) * (dm ** -0.5))
                layer = nn.TransformerEncoderLayer(
                    d_model=dm, nhead=nh, dim_feedforward=int(4 * dm / 3) * 2,
                    dropout=do, batch_first=True, norm_first=True, activation='gelu')
                s.enc = nn.TransformerEncoder(layer, num_layers=nl)
                s.norm = nn.LayerNorm(dm)
                s.head = nn.Sequential(nn.ReLU(), nn.Linear(dm, 1))

            def tokens(s, x):
                h = x.unsqueeze(-1) * s.W + s.b
                return torch.cat([s.cls.expand(len(x), -1, -1), h], dim=1)

            def forward(s, x):
                return s.head(s.norm(s.enc(s.tokens(x))[:, 0])).squeeze(-1)

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
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=1e-5)
        steps = max(1, self.epochs * int(np.ceil(len(ys) / self.batch)))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps)
        Xt = torch.tensor(Xs, device=self.dev_); yt = torch.tensor(ys, device=self.dev_)
        lossf = torch.nn.MSELoss(); self.net_.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(yt), device=self.dev_)
            for i in range(0, len(yt), self.batch):
                idx = perm[i:i + self.batch]
                opt.zero_grad(); lossf(self.net_(Xt[idx]), yt[idx]).backward()
                torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
                opt.step(); sched.step()
        return self

    def _run(self, X, fn):
        import torch
        X = (np.asarray(X, dtype=np.float32) - self.mu_) / self.sd_
        self.net_.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(X), 8192):
                out.append(fn(torch.tensor(X[i:i+8192], device=self.dev_)).cpu().numpy())
        return np.concatenate(out)

    def predict(self, X):
        return self._run(X, self.net_) * self.ys_ + self.ym_

    def transform(self, X):
        """The CLS representation, for use as features downstream."""
        return self._run(X, lambda t: self.net_.norm(self.net_.enc(
            self.net_.tokens(t))[:, 0]))
