# blockforest_real.R -- the published blockForest package, on the folds t71 exported.
#
# The reviewer's objection is that our comparison used our own implementation of
# the Block Forests sampling rules. This runs the CRAN package itself, on the
# identical rows, with the identical seven inputs the panel provides. Feature
# enrichment is Step 1 of the proposed method and is not given to this package.
#
# Blocks: the a-priori semantic grouping used in the paper, contract geometry
# (m, logm, T, K, S, is_call) and the liquidity proxy (spr).
#
# Usage:  Rscript blockforest_real.R [ticker ...]

.libPaths("~/Rlibs")
suppressMessages(library(blockForest))

folds_dir <- path.expand("~/TKDE2026/folds")
out_csv   <- path.expand("~/TKDE2026/results/bf_real.csv")

NUM_TREES     <- 1000     # matches every forest in the paper
NSETS         <- 30       # package default is 300; with 7 inputs in 2 blocks
                          # the per-block mtry space has at most 6 points, so a
                          # coarse tuning grid covers it
NUM_TREES_PRE <- 200      # package default is 1500; reduced for run time

args  <- commandArgs(trailingOnly = TRUE)
ticks <- if (length(args)) args else c("AAPL", "TSLA", "NVDA", "SPY", "QQQ")

files <- list.files(folds_dir, pattern = "_train\\.csv$", full.names = FALSE)
rows  <- list()

for (f in files) {
  stem <- sub("_train\\.csv$", "", f)
  tick <- sub("_.*$", "", stem)
  fold <- sub("^[^_]*_", "", stem)
  if (!(tick %in% ticks)) next

  tr <- read.csv(file.path(folds_dir, paste0(stem, "_train.csv")))
  te <- read.csv(file.path(folds_dir, paste0(stem, "_test.csv")))

  X  <- as.matrix(tr[, setdiff(names(tr), "IV")])
  y  <- tr$IV
  Xt <- as.matrix(te[, setdiff(names(te), "IV")])
  yt <- te$IV

  blocks <- list(1:6, 7)   # contract geometry | liquidity proxy

  t0 <- proc.time()[["elapsed"]]
  fit <- blockfor(X, y, blocks = blocks, block.method = "BlockForest",
                  num.trees = NUM_TREES, nsets = NSETS,
                  num.trees.pre = NUM_TREES_PRE, splitrule = "variance",
                  num.threads = 8, verbose = FALSE)
  p   <- predict(fit$forest, data = Xt)$predictions
  sec <- proc.time()[["elapsed"]] - t0

  r2  <- 1 - sum((yt - p)^2) / sum((yt - mean(yt))^2)
  mae <- mean(abs(yt - p))
  cat(sprintf("%s %s  R2=%.4f  mae=%.4f  (%.0fs)\n", tick, fold, r2, mae, sec))
  flush.console()

  rows[[length(rows) + 1]] <- data.frame(
    tick = tick, fold = fold, model = "BF-package", r2 = r2, mae = mae,
    secs = sec, num_trees = NUM_TREES, nsets = NSETS)
  write.csv(do.call(rbind, rows), out_csv, row.names = FALSE)
}

cat(sprintf("\n%d folds -> %s\n", length(rows), out_csv))
