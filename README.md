# Dynamic Extra Trees

A forest fixes how many features a split may consider, and every convention for
that number counts columns. On an option panel the columns arrive in correlated
groups, so fifteen columns carry about eight independent directions and a node
drawing all fifteen spends seven draws on information it already holds.

**Dynamic Extra Trees counts directions instead.** It groups the feature matrix
into correlated blocks by union-find on the training window and sets the
candidate count to the number of blocks. Nothing is searched, and because the
forest draws `M` of `d` it fits faster than the arm it improves on.

```python
from dynet import DynamicExtraTrees

est = DynamicExtraTrees().fit(X_train, y_train)   # X: the seven base columns
y_hat = est.predict(X_test)
est.M_, est.d_, est.max_features_                 # 8, 15, 0.533
```

This repository holds the reference implementation, the scripts that produce
every number in the paper, and the result files those numbers are read from.
**It does not hold the option panels**: see [Data](#data).

## Install and run

```bash
pip install numpy pandas scipy scikit-learn     # all the rule itself needs
python examples/demo_synthetic.py               # runs with no data of any kind
```

`requirements.txt` adds the libraries the comparators in `experiments/` need:
XGBoost, LightGBM, CatBoost, numba for the Block Forests reference
implementation, and torch, TabPFN and pytorch-tabnet for the deep arms.

## Layout

| path | contents |
|---|---|
| `dynet/dynET.py` | the method: both enrichments, the union-find blocks, the budget, and a scikit-learn estimator |
| `examples/demo_synthetic.py` | block recovery and fit cost on synthetic grouped data, no panel required |
| `examples/schema.md` | the columns the experiment scripts expect, and where to point them |
| `experiments/` | the scripts that produce the paper's tables and figures; they read a licensed panel |
| `results/` | the result files those runs wrote, and the ones the paper's tables and figures are generated from |

The plotting scripts are not included. Every figure in the paper is drawn from
the files in `results/`, so the numbers behind each panel can be read directly
without them.

## The three steps

```python
augment_basic(X)      # 1. seven contract-and-quote columns -> fifteen
block_labels(A)       # 2. union-find on |corr| > 0.75, one label per column
max_features = M / d  # 3. draw M of d candidates at every node
```

Steps 2 and 3 are four lines on top of a standard estimator; the trees are
scikit-learn's `ExtraTreesRegressor` at its own defaults, and nothing in the
library is modified.

## The two enrichments

Step 1 is written twice, and both are in `dynet/dynET.py`. A term is admitted
when it is a product or a ratio of columns the panel already carries, when it is
a quantity a desk reads off the surface, and when a tree would reach it only
through repeated axis-aligned splits.

**Basic feature enrichment**, `augment_basic`, seven columns to fifteen. The
base is moneyness `m = S/K`, its logarithm `k`, the maturity, the strike, the
spot, a call indicator and the bid-ask spread. Eight terms are appended: the
smile-maturity couplings `m*tau` and `k*tau`, the curvature `k^2`, `m^2` and
`tau^2`, and the liquidity couplings `m*spr`, `spr/tau` and `spr*tau`. No
sensitivity and no mid quote enters at any point. Every number in the main paper
is computed on this vector.

**Greek leak-free enrichment**, `base12_leakfree` then `augment_clean18`, twelve
columns to eighteen. This is the twelve-column width of the earlier literature
with the leakage taken out, built in three steps: replace each vendor
sensitivity by the same Black-Scholes function evaluated at a constant
volatility `sigma_0 = 0.30`, which keeps the part that depends on the contract
and drops the part that is the answer; divide out the spot scale, which leaves
each shape a dimensionless function of `(m, tau)` and the call indicator; and
replace the mid quote by the relative spread `spr/S`, the mid being the price
the implied volatility was solved from. Six terms are then appended, the basic
eight without `k*tau` and `spr*tau`, which the base already carries. Applying
`sigma = vega / (S^2 tau gamma)` to the four shapes returns `sigma_0` and
nothing else, so no function of them recovers a contract's own volatility.

A third function, `augment_full`, reproduces the twelve-feature vector of the
earlier literature *with* the vendor sensitivities, so that regime can be
reproduced. The paper's claims are not computed on it.

## Data

**The option panels are OptionMetrics data and cannot be released.** They are
licensed, not public, and the licence does not permit redistribution, so no
panel, no extract and no row-level derivative of one appears in this repository
and none ever will. Readers who want to rerun the experiments need their own
OptionMetrics subscription, through WRDS or directly; the same scripts run on
any export that carries the columns listed in `examples/schema.md`, which also
gives the cleaning rule applied before anything else happens.

`results/` does contain what those runs produced. They are fold-level
summaries, mean `R^2`, MAE and seconds per quarterly window, from which every
table and figure in the paper is generated. No quote, price or row-level value
appears in them.

Two environment variables, both optional:

```bash
export DYNET_DATA=/path/to/panels     # default: ./data, which is gitignored
export DYNET_FIGDIR=/path/to/output   # where a script writes, if it writes
```

`experiments/t4_clean_sweep.py` reads `aapl_2016_2020.csv` from `DYNET_DATA` and
every other script imports `load()` from it; the other panels follow the same
naming, `nvda_2020_2022.csv`, `qqq_2020_2022.csv`, `spy_2020_2022.csv` and
`tsla_2019_2022.csv`.

## Reproducing the paper

With a panel in place:

```bash
python experiments/t24_underlyings.py       # five underlyings, fourteen comparators
python experiments/t18_tuned.py             # every arm searched at an equal budget
python experiments/t59_bf_basic.py          # against Block Forests
python experiments/t55_greek_isolation.py   # the sensitivity identification
python experiments/t56_isolation_audit.py   # the isolation audit, single-threaded
python experiments/t48_clean18.py           # the Greek leak-free enrichment
```

The comparisons added after the first submission:

```bash
python experiments/t63_wing_models.py core AAPL   # per-row predictions, one arm set at a time
python experiments/t65_metrics.py                 # eight error measures from those predictions
python experiments/t72_robust_cells.py            # dependence-robust intervals, cell by cell
python experiments/t73_block_order.py             # blocks ordered by split value, C(m) per fold
python experiments/t75_equal_n.py AAPL            # the rule and the ablation at 10,000 rows
python experiments/t76_tabpfn_equal.py AAPL       # TabPFN on the same rows, separate process
python experiments/t79_enriched_comparators.py    # the boosting libraries on the same 15 columns
python experiments/t80_bf_widths.py               # block samplers on the same trees, 15 and 18
python experiments/t81_export_wide.py folds_wide  # export folds for the R package
Rscript experiments/blockforest_wide.R folds_wide out.csv 15 18
```

The two `blockforest_*.R` scripts need R with the CRAN package `blockForest`
0.2.7; everything else is Python.

## What the result files hold

| file | contents |
|---|---|
| `underlyings.csv` | the five-panel comparison, 48 folds, one row per method and fold |
| `tuned_all.csv` | the same with every setting searched on the training window |
| `sota_*.csv`, `nonparametric_*.csv` | the state-of-the-art and classical surface fits, per panel |
| `overall_by_fold.csv` | pooled `R^2` of every comparator on every fold, from stored predictions |
| `wing_all.csv`, `metrics_all.csv` | the same by moneyness band, eight error measures |
| `robust_cells.csv`, `robust_pvalues.csv` | moving-block bootstrap intervals, pooled and cell by cell |
| `block_order.csv` | blocks ordered by split value per fold, with `C(1)`...`C(15)` |
| `equal_n_*` | the rule, the ablation and TabPFN on identical row counts |
| `enriched_comparators.csv` | the boosting libraries given the same fifteen enriched columns |
| `bf_real.csv`, `bf_package_wide.csv` | the CRAN `blockForest` package, on seven columns and on the enriched matrix |
| `bf_widths.csv` | the block samplers reimplemented on identical Extra Trees, at both widths |
| `rule_timing_30k.csv`, `timing_cpu.csv` | fit and predict seconds per fold |
| `fold_ids.csv` | row counts and SHA digests of each fold, no row-level values |

Two of these files carry more than the paper reports, and it is worth saying so
plainly. `enriched_comparators.csv` gives each boosting library the fifteen
enriched columns, which the paper does not do because the enrichment is Step 1
of the method: the rule still leads there, by `0.020` to `0.037` over 48 folds.
`bf_widths.csv` compares the candidate-sampling rules on one shared Extra Trees
implementation, so that only the sampling differs: the rule leads `BF-varsel`
and `BF-randomblock`, and is level with `BF-default`, whose keep-half-then-mtry
draw lands on a similar expected candidate count. The paper's Block Forests
comparison is against the published package instead, which is what a reader
would install, and reports that comparison at both widths.

Two runtime notes. The gradient-boosting libraries each load their own OpenMP
runtime and the Block Forests reference implementation uses numba's parallel
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

## License

MIT for the code in this repository. The OptionMetrics panels are not covered by
it, are not distributed here, and remain subject to their own licence.
