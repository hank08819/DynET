"""
blocked_et.py -- Extra Trees with pluggable split-candidate sampling.

scikit-learn exposes only a scalar `max_features`, so it cannot express either
the per-node block stratification that Dynamic ET describes or the per-block
`mtry` vector that Block Forests uses.  This implementation makes the sampling
rule a parameter, so the two can be compared on identical trees.

Sampling modes
  0 UNIFORM     draw k candidates uniformly from all d features
                (this is what scikit-learn does, and what the sklearn path of
                 Dynamic ET actually performs)
  1 STRATIFIED  draw exactly one candidate per block
                (what Dynamic ET describes, and what its Mojo path implements)
  2 BLOCKVARSEL draw m_j candidates from block j, with the vector m given
                (Block Forests, BlockVarSel variant)
  3 RANDOMBLOCK choose one block at random, draw all of its features
                (Block Forests, RandomBlock variant)
  4 BLOCKFOREST keep each block with probability 0.5, then BlockVarSel on those
                (Block Forests, default variant)

Everything else follows Extra Trees: no bootstrap, one uniformly random split
threshold per candidate feature, unlimited depth, leaves of size one.
"""
import numpy as np
from numba import njit, prange

@njit(cache=True)
def _candidates(mode, d, blk, n_blocks, k, mvec, seed_state):
    """Return the candidate feature indices for one node under `mode`."""
    out = np.empty(d, np.int64); c = 0
    if mode == 0:
        perm = np.random.permutation(d)
        for i in range(min(k, d)): out[c] = perm[i]; c += 1
    elif mode == 1:
        for b in range(n_blocks):
            idx = np.where(blk == b)[0]
            if idx.size: out[c] = idx[np.random.randint(idx.size)]; c += 1
    elif mode == 2:
        for b in range(n_blocks):
            idx = np.where(blk == b)[0]
            if idx.size == 0: continue
            take = min(mvec[b], idx.size)
            perm = np.random.permutation(idx.size)
            for i in range(take): out[c] = idx[perm[i]]; c += 1
    elif mode == 3:
        b = np.random.randint(n_blocks)
        idx = np.where(blk == b)[0]
        for i in range(idx.size): out[c] = idx[i]; c += 1
    else:
        for b in range(n_blocks):
            if np.random.random() > 0.5: continue
            idx = np.where(blk == b)[0]
            if idx.size == 0: continue
            take = min(mvec[b], idx.size)
            perm = np.random.permutation(idx.size)
            for i in range(take): out[c] = idx[perm[i]]; c += 1
        if c == 0:
            b = np.random.randint(n_blocks); idx = np.where(blk == b)[0]
            for i in range(idx.size): out[c] = idx[i]; c += 1
    return out[:c]

@njit(cache=True)
def _build_tree(X, y, idx, blk, n_blocks, mode, k, mvec, min_leaf, max_nodes):
    feat = np.full(max_nodes, -1, np.int64); thr = np.zeros(max_nodes, np.float64)
    left = np.full(max_nodes, -1, np.int64); right = np.full(max_nodes, -1, np.int64)
    val  = np.zeros(max_nodes, np.float64)
    stack_lo = np.empty(max_nodes, np.int64); stack_hi = np.empty(max_nodes, np.int64)
    stack_nd = np.empty(max_nodes, np.int64)
    order = idx.copy()
    sp = 0; stack_lo[0]=0; stack_hi[0]=order.size; stack_nd[0]=0; sp=1
    n_nodes = 1; d = X.shape[1]
    while sp > 0:
        sp -= 1; lo = stack_lo[sp]; hi = stack_hi[sp]; nd = stack_nd[sp]
        m = 0.0
        for i in range(lo, hi): m += y[order[i]]
        m /= (hi - lo); val[nd] = m
        if hi - lo <= min_leaf or n_nodes + 2 > max_nodes: continue
        cand = _candidates(mode, d, blk, n_blocks, k, mvec, 0)
        best_gain = -1.0; best_f = -1; best_t = 0.0
        tot = 0.0; tot2 = 0.0
        for i in range(lo, hi):
            v = y[order[i]]; tot += v; tot2 += v*v
        n = hi - lo; parent_sse = tot2 - tot*tot/n
        for ci in range(cand.size):
            f = cand[ci]
            vmin = X[order[lo], f]; vmax = vmin
            for i in range(lo+1, hi):
                v = X[order[i], f]
                if v < vmin: vmin = v
                elif v > vmax: vmax = v
            if vmax <= vmin: continue
            t = vmin + np.random.random()*(vmax - vmin)
            sl = 0.0; sl2 = 0.0; nl = 0
            for i in range(lo, hi):
                if X[order[i], f] <= t:
                    v = y[order[i]]; sl += v; sl2 += v*v; nl += 1
            nr = n - nl
            if nl == 0 or nr == 0: continue
            sr = tot - sl; sr2 = tot2 - sl2
            sse = (sl2 - sl*sl/nl) + (sr2 - sr*sr/nr)
            gain = parent_sse - sse
            if gain > best_gain: best_gain = gain; best_f = f; best_t = t
        if best_f < 0: continue
        p = lo
        for i in range(lo, hi):
            if X[order[i], best_f] <= best_t:
                tmp = order[p]; order[p] = order[i]; order[i] = tmp; p += 1
        if p == lo or p == hi: continue
        feat[nd] = best_f; thr[nd] = best_t
        l = n_nodes; r = n_nodes + 1; n_nodes += 2
        left[nd] = l; right[nd] = r
        stack_lo[sp]=lo; stack_hi[sp]=p;  stack_nd[sp]=l; sp+=1
        stack_lo[sp]=p;  stack_hi[sp]=hi; stack_nd[sp]=r; sp+=1
    return feat, thr, left, right, val, n_nodes

@njit(cache=True, parallel=True)
def _predict_forest(Xte, feats, thrs, lefts, rights, vals, n_trees):
    n = Xte.shape[0]; out = np.zeros(n, np.float64)
    for i in prange(n):
        acc = 0.0
        for t in range(n_trees):
            nd = 0
            while lefts[t, nd] >= 0:
                nd = lefts[t, nd] if Xte[i, feats[t, nd]] <= thrs[t, nd] else rights[t, nd]
            acc += vals[t, nd]
        out[i] = acc / n_trees
    return out

class BlockedExtraTrees:
    def __init__(self, n_estimators=300, mode=0, k=None, mvec=None,
                 blocks=None, min_leaf=1, random_state=42):
        self.n_estimators=n_estimators; self.mode=mode; self.k=k
        self.mvec=mvec; self.blocks=blocks; self.min_leaf=min_leaf; self.rs=random_state
    def fit(self, X, y):
        X=np.ascontiguousarray(X,np.float64); y=np.ascontiguousarray(y,np.float64)
        n,d=X.shape
        blk = np.zeros(d,np.int64) if self.blocks is None else np.asarray(self.blocks,np.int64)
        nb = int(blk.max())+1
        k = d if self.k is None else int(self.k)
        mvec = np.ones(nb,np.int64) if self.mvec is None else np.asarray(self.mvec,np.int64)
        max_nodes = max(64, 4*n+2)
        F=[];T=[];L=[];R=[];V=[]
        np.random.seed(self.rs)
        for b in range(self.n_estimators):
            f,t,l,r,v,_ = _build_tree(X,y,np.arange(n),blk,nb,self.mode,k,mvec,
                                      self.min_leaf,max_nodes)
            F.append(f);T.append(t);L.append(l);R.append(r);V.append(v)
        self.F=np.vstack(F);self.T=np.vstack(T);self.L=np.vstack(L)
        self.R=np.vstack(R);self.V=np.vstack(V)
        return self
    def predict(self, X):
        return _predict_forest(np.ascontiguousarray(X,np.float64),
                               self.F,self.T,self.L,self.R,self.V,self.n_estimators)
