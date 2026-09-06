"""seq_baseline.py -- LSTM and GRU on a cross-section, trained properly.

The manuscript's sequence baselines read the d-dimensional feature vector as a
length-d sequence with one channel. That framing is artificial on a cross-section
and is the reason these architectures are not expected to win here. It is not the
reason they scored near zero in the earlier run: that implementation took one
hundred full-batch gradient steps on thirty thousand rows, which is a hundred
updates, and no recurrent network converges on a hundred updates.

This version gives them the same budget the transformer comparators get --
minibatch AdamW under a one-cycle schedule, 120 epochs, batch 512, standardised
inputs and target, gradient clipping, on CPU -- so that whatever they score is a
statement about the architecture on this task rather than about the optimiser.
"""
import numpy as np


class SeqRegressor:
    def __init__(self, kind='lstm', hidden=64, n_layers=2, dropout=0.1,
                 epochs=120, lr=3e-3, batch=512, seed=42):
        self.kind, self.hidden, self.n_layers = kind, hidden, n_layers
        self.dropout, self.epochs, self.lr = dropout, epochs, lr
        self.batch, self.seed = batch, seed

    def _build(self):
        import torch, torch.nn as nn
        Cell = nn.LSTM if self.kind == 'lstm' else nn.GRU

        class Net(nn.Module):
            def __init__(s, hid, nl, do):
                super().__init__()
                s.rnn = Cell(input_size=1, hidden_size=hid, num_layers=nl,
                             batch_first=True, dropout=do if nl > 1 else 0.0)
                s.head = nn.Sequential(nn.LayerNorm(hid), nn.Linear(hid, hid),
                                       nn.GELU(), nn.Linear(hid, 1))

            def forward(s, x):                       # x: (B, d)
                out, _ = s.rnn(x.unsqueeze(-1))      # (B, d, hid)
                return s.head(out[:, -1, :]).squeeze(-1)

        return Net(self.hidden, self.n_layers, self.dropout)

    def fit(self, X, y):
        import torch
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.float32)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-8
        self.ym_, self.ys_ = float(y.mean()), float(y.std() + 1e-8)
        Xs = (X - self.mu_) / self.sd_; ys = (y - self.ym_) / self.ys_
        # LSTM and GRU kernels on Apple's Metal backend abort the process rather
        # than raise, so the recurrent arms run on CPU. The budget is unchanged.
        self.dev_ = torch.device('cpu')
        self.net_ = self._build().to(self.dev_)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.lr,
            total_steps=max(1, self.epochs * int(np.ceil(len(ys) / self.batch))))
        Xt = torch.tensor(Xs, device=self.dev_); yt = torch.tensor(ys, device=self.dev_)
        lossf = torch.nn.SmoothL1Loss()
        self.net_.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(yt), device=self.dev_)
            for i in range(0, len(yt), self.batch):
                idx = perm[i:i + self.batch]
                opt.zero_grad()
                lossf(self.net_(Xt[idx]), yt[idx]).backward()
                torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
                opt.step(); sched.step()
        return self

    def predict(self, X):
        import torch
        X = (np.asarray(X, dtype=np.float32) - self.mu_) / self.sd_
        self.net_.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(X), 8192):
                t = torch.tensor(X[i:i + 8192], device=self.dev_)
                out.append(self.net_(t).cpu().numpy())
        return np.concatenate(out) * self.ys_ + self.ym_
