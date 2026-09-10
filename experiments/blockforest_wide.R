# blockforest_wide.R -- CRAN blockForest on the enriched matrix, at both widths.
#
# The published comparison gives the package the seven columns the panel
# provides. This run hands it Step 1 (the enrichment) and Step 2 (the
# correlation blocks) of the proposed method, so the only remaining difference
# is Step 3: how the budget is spent. The package keeps its own out-of-bag
# tuning of the per-block mtry vector; nothing is tuned for it or against it.
#
# Usage:  Rscript blockforest_wide.R <folds_dir> <out_csv> [width ...]

suppressMessages(library(blockForest))

args      <- commandArgs(trailingOnly = TRUE)
folds_dir <- args[1]
out_csv   <- args[2]
widths    <- if (length(args) > 2) as.integer(args[-(1:2)]) else c(15L, 18L)

NUM_TREES     <- 1000    # matches every forest in the paper
NSETS         <- 30      # as in the seven-column run reported in the paper
NUM_TREES_PRE <- 200

files <- sort(list.files(folds_dir, pattern = "_train\\.csv$"))
rows  <- list()

for (f in files) {
  stem <- sub("_train\\.csv$", "", f)
  w    <- as.integer(sub("^.*_w", "", stem))
  if (!(w %in% widths)) next
  tick <- sub("_.*$", "", stem)
  fold <- sub("^[^_]*_", "", sub("_w[0-9]+$", "", stem))

  tr <- read.csv(file.path(folds_dir, paste0(stem, "_train.csv")))
  te <- read.csv(file.path(folds_dir, paste0(stem, "_test.csv")))
  bl <- scan(file.path(folds_dir, paste0(stem, "_blocks.txt")), quiet = TRUE)

  X  <- as.matrix(tr[, setdiff(names(tr), "IV")]); y  <- tr$IV
  Xt <- as.matrix(te[, setdiff(names(te), "IV")]); yt <- te$IV
  blocks <- split(seq_along(bl), bl)

  t0  <- proc.time()[["elapsed"]]
  fit <- blockfor(X, y, blocks = blocks, block.method = "BlockForest",
                  num.trees = NUM_TREES, nsets = NSETS,
                  num.trees.pre = NUM_TREES_PRE, splitrule = "variance",
                  num.threads = 8, verbose = FALSE)
  p   <- predict(fit$forest, data = Xt)$predictions
  sec <- proc.time()[["elapsed"]] - t0

  r2 <- 1 - sum((yt - p)^2) / sum((yt - mean(yt))^2)
  cat(sprintf("%s %s w%d  R2=%.4f  blocks=%d  (%.0fs)\n",
              tick, fold, w, r2, length(blocks), sec)); flush.console()

  rows[[length(rows) + 1]] <- data.frame(
    tick = tick, fold = fold, width = w, model = "BF-package",
    r2 = r2, mae = mean(abs(yt - p)), secs = sec,
    n_blocks = length(blocks), nsets = NSETS)
  write.csv(do.call(rbind, rows), out_csv, row.names = FALSE)
}
cat(sprintf("\n%d runs -> %s\n", length(rows), out_csv))
