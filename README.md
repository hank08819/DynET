# Greek-Leakage-Isolated Implied Volatility Learning (Dynamic Extra Trees Pricing)

A forest fixes how many features a split may consider, and every convention for
that number counts columns. On an option panel the columns arrive in correlated
groups, so fifteen columns carry about eight independent directions and a node
drawing all fifteen spends seven draws on information it already holds.

**Dynamic Extra Trees counts directions.** It groups the feature matrix into
correlated blocks by union-find on the training window and sets the candidate
count to the number of blocks. Nothing is searched, and because the forest draws
`M` of `d` it fits faster than the arm it improves on.

This repository holds the reference implementation, the scripts that produce
every number in the paper, and the result files those numbers are read from.
**It does not hold the option panels**: see [Data](#data).

```python
from dynet import DynamicExtraTrees

est = DynamicExtraTrees().fit(X_train, y_train)     # X: the seven base columns
y_hat = est.predict(X_test)
est.M_, est.d_, est.max_features_                   # 8, 15, 0.533
```

## Install

```bash
pip install -r requirements.txt      # numpy, pandas, scipy, scikit-learn suffice for the rule
python examples/demo_synthetic.py    # runs with no data of any kind
```

The rule itself needs only NumPy and scikit-learn. The rest of
`requirements.txt` is for the comparators in `experiments/`.

## What is here

| path | contents |
|---|---|
| `dynet/dynET.py` | the method: Step 1 enrichment, Step 2 union-find blocks, Step 3 budget, and a scikit-learn estimator |
| `examples/demo_synthetic.py` | block recovery and fit cost on synthetic grouped data, no panel required |
| `examples/schema.md` | the columns the experiment scripts expect, and where to point them |
| `experiments/` | the scripts that produce the paper's tables and figures; they read a licensed panel |
| `figures/` | the plotting scripts; they read `results/` and write PDFs |
| `results/` | the aggregate result files the paper's tables and figures are generated from: per-fold R², MAE and seconds, nothing row-level |

## The three steps

```python
augment_basic(X)      # 1. seven contract-and-quote columns -> fifteen
block_labels(A)       # 2. union-find on |corr| > 0.75, one label per column
max_features = M / d  # 3. draw M of d candidates at every node
```

Step 1 appends eight interaction terms: two smile-maturity couplings, three
curvature terms and three liquidity couplings, each a product or ratio of
columns the panel already carries. No sensitivity and no mid quote enters.
`augment_clean18` is the second enrichment described in the paper, which keeps
the twelve-column width of the earlier literature without its leakage.

## Data

The panels are **end-of-day option quote files under a commercial licence** and
are not redistributable, so they are not in this repository and never will be.
`examples/schema.md` gives the exact columns and the cleaning rule, so an
equivalent export can be dropped in: set `PANEL` in
`experiments/t4_clean_sweep.py` and every other script follows.

`results/` does contain the outputs those runs produced. They are fold-level
summaries, mean `R^2`, MAE and seconds per quarterly window, from which every
table and figure in the paper is regenerated. No quote, price or row-level value
appears in them.

## Reproducing the paper

With a panel in place:

```bash
python experiments/t24_underlyings.py    # five underlyings, fourteen comparators
python experiments/t18_tuned.py          # every arm searched at an equal budget
python experiments/t59_bf_basic.py       # against Block Forests
python experiments/t55_greek_isolation.py  # the sensitivity identification
python experiments/t56_isolation_audit.py  # the isolation audit, single-threaded
python figures/f5_results.py             # and the other f*.py
```

Two runtime notes. The gradient-boosting libraries each load their own OpenMP
runtime, and the Block Forests reference implementation uses numba's parallel
regions; putting them in one process deadlocks, which is why the Block Forests
and the TabNet/TabPFN experiments run in separate processes. And a forest's
parallel reduction is not bit-reproducible, so the isolation audit runs
single-threaded.

## Citation

> Han, H., & Li, D. Greek-Leakage-Isolated Implied Volatility Learning.

Two results this repository depends on are earlier work:

> Han, H., Forrest, J. Y.-L., Wang, J., Yuan, S., Han, F., & Li, D. (2024).
> Explainable machine learning for high frequency trading dynamics discovery.
> *Information Sciences* 684, 121286.

> Han, F., Ling, Q., Lu, S., & Han, H. (2026). Feature enrichment imitative
> reinforcement learning for high-frequency trading.
> *Expert Systems with Applications* 296, 129043.

## Licence

MIT for the code in this repository. The option panels are not covered by it
and are not distributed here.

## Paths

Two environment variables, both optional:

```bash
export DYNET_DATA=/path/to/panels     # default: ./data, which is gitignored
export DYNET_FIGDIR=/path/to/output   # default: ./figures/out
```

`experiments/t4_clean_sweep.py` reads `aapl_2016_2020.csv` from `DYNET_DATA`,
and the other panels follow the same naming (`nvda_2020_2022.csv`,
`qqq_2020_2022.csv`, `spy_2020_2022.csv`, `tsla_2019_2022.csv`).
